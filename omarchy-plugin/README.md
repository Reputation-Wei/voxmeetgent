# MeetAgent Omarchy / Quickshell 插件

这个插件在 Omarchy 状态栏显示本机 MeetAgent 会议状态和录音时长，并提供本地录音、Gemini Live、停止保存及打开完整 GUI 的快捷操作。它只访问 `127.0.0.1` 上的 MeetAgent GUI；API key 不进入插件文件或 Omarchy 设置。

## 当前功能

- 状态栏显示服务在线状态、录音状态和计时。
- 弹出面板可以设置输出目录和转写语言（自动、中文、英语、德语）。
- 默认只做本地录音和转写。
- 回答方式可切换：语音（需戴耳机，外放时回声会打断 Agent）或文字（不连接 Gemini Live、不上传会议音频，回答显示在面板“Agent 最近回答”和逐字稿中）。默认值读取 `~/.config/meetagent/config.toml` 的 `[live] reply`。
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

修改已安装插件的 QML 后，`rescanPlugins` 不会让 Quickshell 丢弃已缓存的组件，需运行 `omarchy-restart-shell` 才能加载新版本。

## 设置

在 Omarchy 的 bar 设置中可修改：

- `serviceCommand`：面板内“启动 MeetAgent 服务”按钮执行的命令，默认 `meetagent gui --no-open`。从仓库开发时可设为 `uv run --project ~/Projects/voxmeetgent meetagent gui --no-open`。
- `outputDirectory`：默认保存目录。

## 开发状态

已在 Omarchy（Quickshell 0.3.1）上验证：插件加载、面板渲染（使用 Omarchy 的 `Panel`/`KeyboardPanel` 组件并跟随主题）、通过本机服务开始/停止本地录音、状态栏录音计时，以及 Intel Arc B390 上的 Vulkan 转写。Gemini Live 模式未在 Linux 上验收。逐字稿浏览、speaker 校订、打包分发仍待开发。

Quickshell 只能在 Linux/Wayland 下运行，macOS 上无法测试此插件；macOS 上只能测试它调用的本机 HTTP 服务。

## 参考

- [Omarchy shell plugins](https://github.com/omacom/omarchy/blob/quattro/manual/32-shell-plugins.md)
- [Omarchy shell reference](https://github.com/omacom/omarchy/blob/quattro/docs/omarchy-shell.md)
- [Quickshell PopupWindow](https://quickshell.org/docs/v0.3.0/types/Quickshell/PopupWindow/)
