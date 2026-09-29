# MeetAgent Omarchy / Quickshell 插件

这个插件在 Omarchy 状态栏显示本机 MeetAgent 会议状态和录音时长，并提供本地录音、Gemini Live、停止保存及打开完整 GUI 的快捷操作。它只访问 `127.0.0.1` 上的 MeetAgent GUI；API key 不进入插件文件或 Omarchy 设置。

## 当前功能

- 状态栏显示服务在线状态、录音状态和计时。
- 弹出面板可以设置输出目录和转写语言（自动、中文、英语、德语）。
- 默认只做本地录音和转写。
- Gemini Live 需要单独打开，并勾选整场音频上传同意。
- 会后纪要上传有独立同意项，默认关闭。
- 停止会议时调用本机 GUI 的停止接口，让音频和逐字稿正常收尾。
- 可打开完整 Web GUI 查看实时逐字稿、修正说话人和访问会议文件。

## 运行本机服务

在安装了 MeetAgent 的 Omarchy 终端中启动服务：

```sh
meetagent gui --no-open
```

从本仓库开发时可运行：

```sh
uv run meetagent gui --no-open
```

服务仅监听 `127.0.0.1:8765`。保持该进程运行，插件才可操作会议。Live 模式需要在启动服务的进程环境中配置 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`。本地录音与转写不需要 Gemini key。

## 手动安装

在 Omarchy 上，从仓库根目录运行以下命令：

```sh
plugin_dir="$HOME/.config/omarchy/plugins/io.github.voxprogent.meetagent"
mkdir -p "$plugin_dir"
cp omarchy-plugin/manifest.json omarchy-plugin/BarWidget.qml "$plugin_dir/"
omarchy plugin validate "$plugin_dir"
omarchy-shell shell rescanPlugins
omarchy plugin enable io.github.voxprogent.meetagent
omarchy bar move io.github.voxprogent.meetagent --section right
```

如果 Omarchy 插件 CLI 提示命令参数有所不同，以该版本 `omarchy plugin --help` 和官方插件手册为准。移除插件可使用 `omarchy plugin remove io.github.voxprogent.meetagent`，然后删除上面的插件目录。

## 开发状态

目前在 macOS 开发环境完成插件 manifest、QML 和 API 对接。尚未在 Omarchy/Quickshell 主机执行插件验证或 GUI/麦克风验收。首版弹出面板提供快速控制；逐字稿浏览、speaker 校订、状态服务自动启动、打包分发仍待开发。

## 参考

- [Omarchy shell plugins](https://github.com/omacom/omarchy/blob/quattro/manual/32-shell-plugins.md)
- [Omarchy shell reference](https://github.com/omacom/omarchy/blob/quattro/docs/omarchy-shell.md)
- [Quickshell PopupWindow](https://quickshell.org/docs/v0.3.0/types/Quickshell/PopupWindow/)
