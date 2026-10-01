#!/usr/bin/env bash
# Rebuild pywhispercpp in the project venv with the ggml Vulkan backend.
#
# The PyPI wheel is CPU-only. On Intel Arc (e.g. Panther Lake B390) the Vulkan
# build transcribes roughly 5x faster with large-v3-turbo.
#
# Needs: vulkan-icd-loader + a Vulkan driver (vulkan-intel), shaderc (glslc),
# cmake, gcc. Vulkan-Headers and SPIRV-Headers are used from the system when
# installed (pacman -S vulkan-headers spirv-headers); otherwise they are fetched
# into a cache directory so no root access is required.
#
# Rerun after `uv sync --reinstall` or recreating .venv.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_root/.venv/bin/python"
cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/meetagent/vulkan-sdk"
prefix="$cache_dir/prefix"
vulkan_headers_tag="v1.4.357"
spirv_headers_tag="vulkan-sdk-1.4.335.0"
pywhispercpp_version="$("$python" -c 'import importlib.metadata as m; print(m.version("pywhispercpp"))')"

for tool in glslc cmake git uv; do
  command -v "$tool" >/dev/null || { echo "缺少 $tool" >&2; exit 1; }
done

fetch_headers() {
  local name="$1" tag="$2"
  if [[ ! -d "$cache_dir/$name" ]]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$tag" \
      "https://github.com/KhronosGroup/$name.git" "$cache_dir/$name"
  fi
  cmake -S "$cache_dir/$name" -B "$cache_dir/$name/build" -DCMAKE_INSTALL_PREFIX="$prefix" >/dev/null
  cmake --install "$cache_dir/$name/build" >/dev/null
}

prefix_path=""
if [[ ! -f /usr/include/vulkan/vulkan.h || ! -d /usr/include/spirv ]]; then
  mkdir -p "$cache_dir"
  fetch_headers Vulkan-Headers "$vulkan_headers_tag"
  fetch_headers SPIRV-Headers "$spirv_headers_tag"
  prefix_path="$prefix"
fi

# pywhispercpp's setup.py forwards every environment variable to CMake as -D,
# so build with a minimal environment instead of the user's full shell.
env -i HOME="$HOME" PATH=/usr/local/bin:/usr/bin:/bin \
  GGML_VULKAN=1 GGML_CCACHE=OFF \
  CMAKE_PREFIX_PATH="$prefix_path" \
  CMAKE_BUILD_WITH_INSTALL_RPATH=ON 'CMAKE_INSTALL_RPATH=$ORIGIN' \
  CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  uv pip install --python "$python" --no-binary pywhispercpp \
    --reinstall-package pywhispercpp --no-cache "pywhispercpp==$pywhispercpp_version"

"$python" - <<'EOF'
import importlib.util, pathlib
origin = pathlib.Path(importlib.util.find_spec("_pywhispercpp").origin).parent
assert any(origin.glob("libggml-vulkan.so*")), "Vulkan 后端未生成"
print("pywhispercpp Vulkan 后端已安装")
EOF
