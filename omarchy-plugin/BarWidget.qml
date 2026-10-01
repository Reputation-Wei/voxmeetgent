import QtQuick
import Quickshell
import qs.Commons
import qs.Ui

// Omarchy bar widget for the local MeetAgent service (127.0.0.1:8765). The bar
// button shows service/recording state; its popup starts and stops meetings.
// The plugin never sees the Gemini key: Live mode only works when the service
// process itself has GEMINI_API_KEY or GOOGLE_API_KEY.
Panel {
  id: root
  moduleName: "io.github.voxprogent.meetagent"
  ipcTarget: "io.github.voxprogent.meetagent"

  readonly property string apiBase: "http://127.0.0.1:" + setting("port", 8765)
  readonly property string serviceCommand: setting("serviceCommand", "meetagent gui --no-open")

  property string token: ""
  // Not `state`: that would shadow QtQuick's Item.state.
  property string meetingState: "offline"
  property string meetingId: ""
  property int elapsedSeconds: 0
  property string outputDirectory: setting("outputDirectory", "~/Meetings")
  property string language: setting("language", "auto")
  property string summaryLanguage: setting("summaryLanguage", "zh")
  property string errorText: ""
  property bool busy: false
  property bool serviceStarting: false
  property bool liveMode: false
  // "voice" speaks answers (needs headphones: speaker echo cuts the Agent off);
  // "text" only shows them and never connects Gemini Live.
  property string replyMode: "voice"
  property bool replyChosen: false
  property string lastAgentText: ""
  property string lastAgentTime: ""
  property bool consentToStream: false
  property bool consentToSummaryUpload: false

  readonly property bool active: ["recording", "starting", "stopping"].indexOf(meetingState) !== -1
  readonly property color foreground: bar ? bar.foreground : Color.foreground
  readonly property color urgent: bar ? bar.urgent : Color.urgent
  readonly property color dim: Qt.darker(foreground, 1.55)
  readonly property string fontFamily: bar ? bar.fontFamily : Style.font.family

  function request(method, path, body, callback) {
    var xhr = new XMLHttpRequest()
    xhr.open(method, apiBase + path)
    xhr.setRequestHeader("Content-Type", "application/json")
    if (token.length > 0)
      xhr.setRequestHeader("X-MeetAgent-Token", token)
    xhr.onreadystatechange = function() {
      if (xhr.readyState !== XMLHttpRequest.DONE)
        return
      // Qt reports connection failures as status 0 rather than via onerror.
      if (xhr.status === 0) {
        meetingState = "offline"
        if (callback) callback(false, null)
        return
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        var detail = "本机 MeetAgent 服务请求失败 (" + xhr.status + ")"
        try { detail = JSON.parse(xhr.responseText).detail || detail } catch (e) {}
        errorText = String(detail)
        if (xhr.status === 401) token = ""
        if (callback) callback(false, null)
        return
      }
      var result = {}
      try { result = xhr.responseText.length ? JSON.parse(xhr.responseText) : {} } catch (e) {}
      if (callback) callback(true, result)
    }
    xhr.send(body === null ? null : JSON.stringify(body))
  }

  function refreshStatus() {
    if (token.length === 0) {
      request("GET", "/api/bootstrap", null, function(ok, result) {
        if (!ok) return
        token = result.token || ""
        loadStatus()
      })
      return
    }
    loadStatus()
  }

  function loadStatus() {
    request("GET", "/api/status", null, function(ok, result) {
      if (!ok) return
      serviceStarting = false
      meetingState = result.state || "idle"
      meetingId = result.meeting_id || ""
      elapsedSeconds = result.elapsed_seconds || 0
      // Follow the configured default until the user picks a mode here.
      if (!replyChosen && !active && result.reply) replyMode = result.reply
      lastAgentText = result.last_agent ? String(result.last_agent.text || "") : ""
      lastAgentTime = result.last_agent ? String(result.last_agent.time || "") : ""
    })
  }

  readonly property string serviceLog: "~/.cache/meetagent/service.log"

  function startService() {
    if (!bar) return
    errorText = ""
    serviceStarting = true
    // bar.run is fire-and-forget, so keep the output where a failure can be read.
    bar.run("mkdir -p ~/.cache/meetagent && (" + serviceCommand + ") >> " + serviceLog + " 2>&1")
    serviceStartTimeout.restart()
  }

  Timer {
    id: serviceStartTimeout
    interval: 20000
    onTriggered: {
      if (root.meetingState !== "offline") return
      root.serviceStarting = false
      root.errorText = "20 秒内没有连上服务。请检查启动命令“" + root.serviceCommand
        + "”是否可在终端运行，并查看 " + root.serviceLog
    }
  }

  function startMeeting(live) {
    if (busy || active) return
    if (live && !consentToStream) {
      errorText = "启用 Gemini Live 前，请确认与会者同意整场音频上传。"
      return
    }
    errorText = ""
    busy = true
    request("POST", "/api/meetings", {
      output_dir: outputDirectory,
      language: language,
      summary_language: summaryLanguage,
      live: live,
      consent_to_stream: live && consentToStream,
      consent_to_summary_upload: live && consentToSummaryUpload,
      reply: replyMode
    }, function(ok, result) {
      busy = false
      if (ok) {
        meetingId = result.meeting_id || ""
        refreshStatus()
      }
    })
  }

  function stopMeeting() {
    if (busy || meetingId.length === 0) return
    busy = true
    request("POST", "/api/meetings/" + encodeURIComponent(meetingId) + "/stop", {}, function(ok) {
      busy = false
      if (ok) refreshStatus()
    })
  }

  function openGui() {
    if (bar) bar.run("xdg-open " + apiBase + "/")
    close()
  }

  function clock(secs) {
    var h = Math.floor(secs / 3600), m = Math.floor(secs / 60) % 60, s = secs % 60
    var mm = (m < 10 ? "0" : "") + m, ss = (s < 10 ? "0" : "") + s
    return h > 0 ? h + ":" + mm + ":" + ss : mm + ":" + ss
  }

  function stateLabel() {
    if (meetingState === "recording") return "录音 " + clock(elapsedSeconds)
    if (meetingState === "stopping") return "正在保存"
    if (meetingState === "starting") return "正在启动"
    if (meetingState === "error") return "会议错误"
    if (meetingState === "offline") return serviceStarting ? "服务启动中" : "服务未运行"
    return "空闲"
  }

  onOpenedChanged: if (opened) refreshStatus()

  // Poll faster while a meeting runs so the bar clock keeps up.
  Timer {
    interval: root.meetingState === "recording" ? 1000 : 3000
    repeat: true
    running: true
    triggeredOnStart: true
    onTriggered: root.refreshStatus()
  }

  // WidgetButton (not BarIconButton, which has a fixed icon-slot width) so the
  // button grows to fit the recording clock instead of overlapping neighbours.
  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.meetingState === "recording" ? "● " + root.clock(root.elapsedSeconds)
        : root.meetingState === "stopping" ? "● 保存中"
        : "◉"
    active: root.active
    dimmed: root.meetingState === "offline"
    tooltipText: ""
    onPressed: function(buttonCode) {
      if (buttonCode === Qt.MiddleButton) root.refreshStatus()
      else root.toggle()
    }
  }

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(360))
    contentHeight: panel.fittedContentHeight(column.implicitHeight, Style.space(760))

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }

      Column {
        id: column
        width: parent.width
        spacing: Style.space(12)

        PanelHero {
          width: parent.width
          title: "MeetAgent"
          meta: root.stateLabel()
          foreground: root.foreground
          fontFamily: root.fontFamily
        }

        Text {
          width: parent.width
          textFormat: Text.PlainText
          text: root.meetingState === "recording" ? "正在本机保存音频和逐字稿"
              : root.meetingState === "offline" ? "插件只连接本机 127.0.0.1 上的 MeetAgent 服务"
              : "会议音频与逐字稿保存在本机"
          color: root.dim
          font.family: root.fontFamily
          font.pixelSize: Style.font.bodySmall
          wrapMode: Text.WordWrap
        }

        Button {
          visible: root.meetingState === "offline"
          width: parent.width
          text: root.serviceStarting ? "正在启动服务…" : "启动 MeetAgent 服务"
          foreground: root.foreground
          fontFamily: root.fontFamily
          bordered: true
          enabled: !root.serviceStarting
          onClicked: root.startService()
        }

        Column {
          visible: root.meetingState !== "offline"
          width: parent.width
          spacing: Style.space(12)

          PanelSeparator { foreground: root.foreground }

          PanelSectionHeader {
            text: "保存目录"
            foreground: root.foreground
            fontFamily: root.fontFamily
          }
          TextField {
            width: parent.width
            text: root.outputDirectory
            placeholderText: "~/Meetings"
            foreground: root.foreground
            enabled: !root.active
            onTextEdited: root.outputDirectory = text
          }

          PanelSectionHeader {
            text: "转写语言"
            foreground: root.foreground
            fontFamily: root.fontFamily
          }
          ButtonGroup {
            options: [
              { value: "auto", label: "自动" },
              { value: "zh", label: "中文" },
              { value: "en", label: "English" },
              { value: "de", label: "Deutsch" }
            ]
            value: root.language
            foreground: root.foreground
            fontFamily: root.fontFamily
            focusable: false
            enabled: !root.active
            onChanged: function(value) { root.language = value }
          }

          PanelSectionHeader {
            text: "回答方式"
            foreground: root.foreground
            fontFamily: root.fontFamily
          }
          ButtonGroup {
            options: [
              { value: "voice", label: "语音" },
              { value: "text", label: "文字" }
            ]
            value: root.replyMode
            foreground: root.foreground
            fontFamily: root.fontFamily
            focusable: false
            enabled: !root.active
            onChanged: function(value) {
              root.replyMode = value
              root.replyChosen = true
              if (value === "text") root.liveMode = false
            }
          }
          Text {
            width: parent.width
            textFormat: Text.PlainText
            text: root.replyMode === "text"
              ? "回答只显示在下方和逐字稿中，不连接 Gemini Live、不上传会议音频。"
              : "语音播报请戴耳机：外放时麦克风会录到 Agent 的声音并把它打断。"
            color: root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
            wrapMode: Text.WordWrap
          }

          PanelSeparator { foreground: root.foreground }

          Toggle {
            visible: root.replyMode === "voice"
            width: parent.width
            label: "Gemini Live"
            description: "实时语音 Agent；整场音频会发送给 Gemini"
            checked: root.liveMode
            foreground: root.foreground
            fontFamily: root.fontFamily
            enabled: !root.active
            onClicked: root.liveMode = !root.liveMode
          }
          Toggle {
            visible: root.liveMode
            width: parent.width
            label: "与会者已同意音频上传"
            checked: root.consentToStream
            foreground: root.foreground
            fontFamily: root.fontFamily
            enabled: !root.active
            onClicked: root.consentToStream = !root.consentToStream
          }
          Toggle {
            visible: root.liveMode
            width: parent.width
            label: "允许上传逐字稿生成纪要"
            checked: root.consentToSummaryUpload
            foreground: root.foreground
            fontFamily: root.fontFamily
            enabled: !root.active
            onClicked: root.consentToSummaryUpload = !root.consentToSummaryUpload
          }

          Row {
            width: parent.width
            spacing: Style.space(8)

            Button {
              width: (parent.width - parent.spacing) / 2
              text: root.active ? "停止并保存" : (root.liveMode && root.replyMode === "voice" ? "开始 Live 会议" : "开始录音")
              foreground: root.active ? root.urgent : root.foreground
              fontFamily: root.fontFamily
              bordered: true
              enabled: !root.busy && (root.active ? root.meetingState === "recording" : true)
              onClicked: root.active ? root.stopMeeting() : root.startMeeting(root.liveMode && root.replyMode === "voice")
            }
            Button {
              width: (parent.width - parent.spacing) / 2
              text: "打开完整 GUI"
              foreground: root.foreground
              fontFamily: root.fontFamily
              bordered: true
              onClicked: root.openGui()
            }
          }
        }

        Column {
          visible: root.lastAgentText.length > 0
          width: parent.width
          spacing: Style.space(6)

          PanelSeparator { foreground: root.foreground }
          PanelSectionHeader {
            text: "Agent 最近回答" + (root.lastAgentTime ? " · " + root.lastAgentTime : "")
            foreground: root.foreground
            fontFamily: root.fontFamily
          }
          Text {
            width: parent.width
            textFormat: Text.PlainText
            // Analyst answers are Markdown; strip heading/list markers for the panel.
            text: root.lastAgentText.replace(/^#+\s*/gm, "").replace(/\*\*/g, "")
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
            wrapMode: Text.WordWrap
            maximumLineCount: 14
            elide: Text.ElideRight
          }
        }

        Text {
          width: parent.width
          visible: root.errorText.length > 0
          textFormat: Text.PlainText
          text: root.errorText
          color: root.urgent
          font.family: root.fontFamily
          font.pixelSize: Style.font.bodySmall
          wrapMode: Text.WordWrap
        }
      }
    }
  }
}
