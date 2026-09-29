import QtQuick
import QtQuick.Controls
import Quickshell

Item {
    id: root
    implicitWidth: badge.implicitWidth + 20
    implicitHeight: 32

    // Omarchy supplies `bar`, `moduleName` and `settings` to bar widgets.
    property var bar
    property string moduleName: "meetagent"
    property var settings

    property string apiBase: "http://127.0.0.1:8765"
    property string token: ""
    property string state: "offline"
    property string meetingId: ""
    property int elapsedSeconds: 0
    property string outputDirectory: "~/Meetings"
    property string language: "auto"
    property string summaryLanguage: "zh"
    property string errorText: ""
    property bool busy: false
    property bool panelOpen: false
    property bool liveMode: false
    property bool consentToStream: false
    property bool consentToSummaryUpload: false

    function request(method, path, body, callback) {
        const xhr = new XMLHttpRequest()
        xhr.open(method, apiBase + path)
        xhr.setRequestHeader("Content-Type", "application/json")
        if (token.length > 0)
            xhr.setRequestHeader("X-MeetAgent-Token", token)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE)
                return
            if (xhr.status < 200 || xhr.status >= 300) {
                let detail = "本机 MeetAgent 服务请求失败 (" + xhr.status + ")"
                try { detail = JSON.parse(xhr.responseText).detail || detail } catch (e) {}
                errorText = detail
                if (xhr.status === 401) token = ""
                if (callback) callback(false, null)
                return
            }
            errorText = ""
            let result = {}
            try { result = xhr.responseText.length ? JSON.parse(xhr.responseText) : {} } catch (e) {}
            if (callback) callback(true, result)
        }
        xhr.onerror = function() {
            errorText = "MeetAgent GUI 未运行。请先启动本机 GUI 服务。"
            if (callback) callback(false, null)
        }
        xhr.send(body === null ? null : JSON.stringify(body))
    }

    function refreshStatus() {
        if (token.length === 0) {
            request("GET", "/api/bootstrap", null, function(ok, result) {
                if (!ok) {
                    state = "offline"
                    return
                }
                token = result.token || ""
                loadStatus()
            })
            return
        }
        loadStatus()
    }

    function loadStatus() {
        request("GET", "/api/status", null, function(ok, result) {
            if (!ok) {
                state = "offline"
                return
            }
            state = result.state || "idle"
            meetingId = result.meeting_id || ""
            elapsedSeconds = result.elapsed_seconds || 0
        })
    }

    function startMeeting() {
        if (busy) return
        if (liveMode && !consentToStream) {
            errorText = "启用 Gemini Live 前，请确认与会者同意整场音频上传。"
            return
        }
        busy = true
        request("POST", "/api/meetings", {
            output_dir: outputDirectory,
            language: language,
            summary_language: summaryLanguage,
            live: liveMode,
            consent_to_stream: liveMode && consentToStream,
            consent_to_summary_upload: liveMode && consentToSummaryUpload
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
        request("POST", "/api/meetings/" + encodeURIComponent(meetingId) + "/stop", {}, function(ok, result) {
            busy = false
            if (ok) refreshStatus()
        })
    }

    function openGui() {
        if (root.bar && root.bar.run)
            root.bar.run("xdg-open http://127.0.0.1:8765/")
        else
            errorText = "请在浏览器打开 http://127.0.0.1:8765/"
    }

    function elapsedLabel() {
        const minutes = Math.floor(elapsedSeconds / 60).toString().padStart(2, "0")
        const seconds = (elapsedSeconds % 60).toString().padStart(2, "0")
        return minutes + ":" + seconds
    }

    function stateLabel() {
        if (state === "recording") return "录音 " + elapsedLabel()
        if (state === "stopping") return "正在保存"
        if (state === "starting") return "正在启动"
        if (state === "error") return "会议错误"
        if (state === "offline") return "MeetAgent 离线"
        return "MeetAgent"
    }

    Rectangle {
        id: badge
        anchors.centerIn: parent
        implicitWidth: badgeText.implicitWidth + 22
        implicitHeight: 26
        radius: 7
        color: root.state === "recording" ? "#7f1d1d" : root.state === "offline" ? "#27272a" : "#27272a"
        border.width: root.state === "recording" ? 1 : 0
        border.color: "#f87171"

        Text {
            id: badgeText
            anchors.centerIn: parent
            text: (root.state === "recording" ? "●  " : "◉  ") + root.stateLabel()
            color: root.state === "recording" ? "#fecaca" : "#e4e4e7"
            font.pixelSize: 12
        }
        MouseArea {
            anchors.fill: parent
            onClicked: root.panelOpen = !root.panelOpen
            cursorShape: Qt.PointingHandCursor
        }
    }

    Timer {
        interval: 1500
        repeat: true
        running: true
        onTriggered: root.refreshStatus()
    }

    PopupWindow {
        id: panel
        anchor.item: root
        width: 360
        height: root.liveMode ? 438 : 360
        visible: root.panelOpen
        color: "transparent"
        grabFocus: true
        onVisibleChanged: if (visible) root.refreshStatus()

        Rectangle {
            anchors.fill: parent
            radius: 12
            color: "#18181b"
            border.color: "#3f3f46"
            border.width: 1

            Column {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10

                Row {
                    width: parent.width
                    spacing: 8
                    Text { text: "MeetAgent"; color: "#fafafa"; font.pixelSize: 18; font.bold: true }
                    Item { width: 1; height: 1 }
                    Text { text: root.stateLabel(); color: root.state === "recording" ? "#fca5a5" : "#a1a1aa"; font.pixelSize: 12 }
                }

                Text {
                    width: parent.width
                    text: root.state === "recording" ? "正在本机保存音频和逐字稿 · " + root.elapsedLabel() : "会议音频与逐字稿保存在本机。"
                    color: "#d4d4d8"
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                }

                Text { text: "保存目录"; color: "#a1a1aa"; font.pixelSize: 11 }
                TextField {
                    id: outputField
                    width: parent.width
                    height: 34
                    text: root.outputDirectory
                    color: "#fafafa"
                    placeholderText: "~/Meetings"
                    onTextEdited: root.outputDirectory = text
                    background: Rectangle { color: "#27272a"; radius: 6; border.color: "#52525b" }
                }

                Row {
                    width: parent.width
                    spacing: 8
                    Text { text: "转写语言"; color: "#a1a1aa"; anchors.verticalCenter: parent.verticalCenter; font.pixelSize: 11 }
                    ComboBox {
                        id: languageBox
                        width: 110
                        height: 30
                        model: ["自动", "中文", "English", "Deutsch"]
                        currentIndex: 0
                        onActivated: root.language = ["auto", "zh", "en", "de"][currentIndex]
                    }
                    Text { text: "默认本地模式"; color: "#a1a1aa"; anchors.verticalCenter: parent.verticalCenter; font.pixelSize: 11 }
                }

                CheckBox {
                    id: liveCheck
                    text: "启用 Gemini Live（实时语音 Agent）"
                    checked: root.liveMode
                    onToggled: root.liveMode = checked
                    contentItem: Text { text: liveCheck.text; color: "#e4e4e7"; leftPadding: liveCheck.indicator.width + liveCheck.spacing; verticalAlignment: Text.AlignVCenter; font.pixelSize: 12 }
                }
                CheckBox {
                    id: streamConsent
                    visible: root.liveMode
                    text: "我已取得同意：整场音频会发送给 Gemini"
                    checked: root.consentToStream
                    onToggled: root.consentToStream = checked
                    contentItem: Text { text: streamConsent.text; color: "#fca5a5"; leftPadding: streamConsent.indicator.width + streamConsent.spacing; verticalAlignment: Text.AlignVCenter; wrapMode: Text.WordWrap; font.pixelSize: 11 }
                }
                CheckBox {
                    id: summaryConsent
                    visible: root.liveMode
                    text: "允许将逐字稿发送给 Gemini 生成纪要"
                    checked: root.consentToSummaryUpload
                    onToggled: root.consentToSummaryUpload = checked
                    contentItem: Text { text: summaryConsent.text; color: "#d4d4d8"; leftPadding: summaryConsent.indicator.width + summaryConsent.spacing; verticalAlignment: Text.AlignVCenter; wrapMode: Text.WordWrap; font.pixelSize: 11 }
                }

                Row {
                    width: parent.width
                    spacing: 8
                    Rectangle {
                        width: (parent.width - 8) / 2
                        height: 36
                        radius: 7
                        color: root.busy || ["recording", "starting", "stopping"].includes(root.state) ? "#52525b" : "#166534"
                        Text { anchors.centerIn: parent; text: "开始本地录音"; color: "white"; font.pixelSize: 12 }
                        MouseArea { anchors.fill: parent; enabled: !root.busy && !["recording", "starting", "stopping"].includes(root.state); onClicked: { root.liveMode = false; root.startMeeting() } }
                    }
                    Rectangle {
                        width: (parent.width - 8) / 2
                        height: 36
                        radius: 7
                        color: root.state === "recording" ? "#991b1b" : "#3f3f46"
                        Text { anchors.centerIn: parent; text: "停止并保存"; color: "white"; font.pixelSize: 12 }
                        MouseArea { anchors.fill: parent; enabled: !root.busy && ["recording", "starting", "stopping"].includes(root.state); onClicked: root.stopMeeting() }
                    }
                }

                Row {
                    width: parent.width
                    spacing: 8
                    Rectangle {
                        width: (parent.width - 8) / 2
                        height: 32
                        radius: 6
                        color: "#27272a"
                        Text { anchors.centerIn: parent; text: "开始 Live 会议"; color: "#e4e4e7"; font.pixelSize: 11 }
                        MouseArea { anchors.fill: parent; enabled: root.liveMode && !root.busy && !["recording", "starting", "stopping"].includes(root.state); onClicked: root.startMeeting() }
                    }
                    Rectangle {
                        width: (parent.width - 8) / 2
                        height: 32
                        radius: 6
                        color: "#27272a"
                        Text { anchors.centerIn: parent; text: "打开完整 GUI"; color: "#e4e4e7"; font.pixelSize: 11 }
                        MouseArea { anchors.fill: parent; onClicked: root.openGui() }
                    }
                }

                Text {
                    width: parent.width
                    visible: root.errorText.length > 0
                    text: root.errorText
                    color: "#fca5a5"
                    wrapMode: Text.WordWrap
                    font.pixelSize: 11
                }
            }
        }
    }
}
