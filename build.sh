#!/usr/bin/env bash
# ==============================================================================
#  ViRheometry — 環境セットアップスクリプト
#
#  対象 OS  : macOS (Apple Silicon / Intel)  |  Ubuntu/Debian Linux
#  Python   : 3.11 推奨 (3.10 以上必須)
#  実行場所 : プロジェクトルート (build.sh があるディレクトリ)
#
#  使い方:
#    chmod +x build.sh
#    ./build.sh              # フルセットアップ
#    ./build.sh --py-only    # Python 環境のみ
#    ./build.sh --cpp-only   # C++ ビルドのみ
# ==============================================================================

set -euo pipefail

# ── 色付きログ ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_section() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BOLD}${BLUE}  $1${NC}"; echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }
log_ok()      { echo -e "  ${GREEN}✓${NC}  $1"; }
log_warn()    { echo -e "  ${YELLOW}⚠${NC}  $1"; }
log_error()   { echo -e "  ${RED}✗${NC}  $1" >&2; }
log_info()    { echo -e "  ${CYAN}→${NC}  $1"; }
die()         { log_error "$1"; exit 1; }

# ── 引数解析 ──────────────────────────────────────────────────────────────────
DO_PYTHON=true
DO_CPP=true

for arg in "$@"; do
  case $arg in
    --py-only)   DO_CPP=false ;;
    --cpp-only)  DO_PYTHON=false ;;
    --help|-h)
      echo "使い方: ./build.sh [--py-only] [--cpp-only]"
      exit 0 ;;
    *) log_warn "不明なオプション: $arg (無視します)" ;;
  esac
done

# ── OS / アーキテクチャ検出 ───────────────────────────────────────────────────
log_section "システム環境を確認"

OS="$(uname -s)"
ARCH="$(uname -m)"
log_info "OS: ${OS}  /  Arch: ${ARCH}"

IS_MACOS=false
IS_LINUX=false
HOMEBREW_PREFIX=""

case "$OS" in
  Darwin)
    IS_MACOS=true
    if [[ "$ARCH" == "arm64" ]]; then
      HOMEBREW_PREFIX="/opt/homebrew"
      log_info "Apple Silicon Mac を検出"
    else
      HOMEBREW_PREFIX="/usr/local"
      log_info "Intel Mac を検出"
    fi
    ;;
  Linux)
    IS_LINUX=true
    log_info "Linux を検出"
    ;;
  *)
    die "非対応のOS: ${OS}。macOS または Linux が必要です。"
    ;;
esac

# プロジェクトルート = このスクリプトのディレクトリ
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_info "プロジェクトルート: ${PROJECT_ROOT}"

# ==============================================================================
#  STEP 1: システム依存関係のインストール
# ==============================================================================
log_section "STEP 1: システムパッケージ"

if $IS_MACOS; then
  # ── Homebrew ──────────────────────────────────────────────────────────────
  if ! command -v brew &>/dev/null; then
    log_info "Homebrew が見つかりません。インストールします..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Apple Silicon の場合は PATH を通す
    if [[ "$ARCH" == "arm64" ]]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
  fi
  log_ok "Homebrew: $(brew --version | head -1)"

  # Xcode Command Line Tools (OpenGL/GLUT はここに含まれる)
  if ! xcode-select -p &>/dev/null; then
    log_info "Xcode Command Line Tools をインストールします..."
    xcode-select --install
    echo "  インストール完了後、このスクリプトを再実行してください。"
    exit 0
  fi
  log_ok "Xcode CLT: $(xcode-select -p)"

  # Homebrew パッケージ
  BREW_PACKAGES=("cmake" "eigen" "opencv" "python@3.11")
  for pkg in "${BREW_PACKAGES[@]}"; do
    if brew list "$pkg" &>/dev/null; then
      log_ok "${pkg}: インストール済み"
    else
      log_info "${pkg} をインストール中..."
      brew install "$pkg"
      log_ok "${pkg}: インストール完了"
    fi
  done

  # Python 3.11 を優先 PATH に追加
  export PATH="${HOMEBREW_PREFIX}/opt/python@3.11/bin:${PATH}"

elif $IS_LINUX; then
  # ── apt (Ubuntu / Debian) ──────────────────────────────────────────────────
  if ! command -v apt-get &>/dev/null; then
    die "apt-get が見つかりません。Ubuntu/Debian 以外の Linux は手動でパッケージをインストールしてください。"
  fi

  log_info "パッケージリストを更新中..."
  sudo apt-get update -qq

  APT_PACKAGES=(
    "cmake" "build-essential" "git"
    "libeigen3-dev"
    "libopencv-dev"
    "freeglut3-dev" "libgl1-mesa-dev" "libglu1-mesa-dev"
    "python3.11" "python3.11-dev" "python3.11-venv" "python3-pip"
  )

  for pkg in "${APT_PACKAGES[@]}"; do
    if dpkg -l "$pkg" &>/dev/null; then
      log_ok "${pkg}: インストール済み"
    else
      log_info "${pkg} をインストール中..."
      sudo apt-get install -y -qq "$pkg"
      log_ok "${pkg}: インストール完了"
    fi
  done
fi

# ==============================================================================
#  STEP 2: libcmaes (Python バインディング付き C++ ライブラリ)
# ==============================================================================
log_section "STEP 2: libcmaes (CMA-ES C++ ライブラリ)"

LCMAES_OK=false

if python3 -c "import lcmaes" &>/dev/null 2>&1; then
  log_ok "lcmaes: インストール済み"
  LCMAES_OK=true
else
  log_info "lcmaes をソースからビルドします (github.com/beniz/libcmaes)..."

  LCMAES_BUILD_DIR="/tmp/libcmaes_build"
  mkdir -p "$LCMAES_BUILD_DIR"

  if ! command -v git &>/dev/null; then
    log_warn "git が見つかりません。lcmaes のビルドをスキップします。"
    log_warn "  → Optimization/libs/cmaes.py の lcmaes 依存機能は使用不可になります。"
  else
    (
      cd "$LCMAES_BUILD_DIR"
      if [ ! -d "libcmaes" ]; then
        git clone --depth=1 https://github.com/beniz/libcmaes.git
      fi
      cd libcmaes
      mkdir -p build && cd build

      # Eigen5 互換: バージョン制約を除去
      sed -i.bak 's/find_package(Eigen3 [0-9.]*)/find_package(Eigen3)/' CMakeLists.txt 2>/dev/null || true
      sed -i.bak 's/find_package(eigen3 [0-9.]*)/find_package(Eigen3)/' CMakeLists.txt 2>/dev/null || true

      if $IS_MACOS; then
        cmake .. \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBCMAES_BUILD_PYTHON=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DEIGEN3_INCLUDE_DIR="${HOMEBREW_PREFIX}/include/eigen3"
      else
        cmake .. \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBCMAES_BUILD_PYTHON=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local
      fi

      make -j"$(nproc 2>/dev/null || sysctl -n hw.physicalcpu)"
      sudo make install

      # Python バインディングを site-packages に配置
      if [ -f python/lcmaes.so ] || [ -f python/lcmaes.cpython-*.so ]; then
        SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
        sudo cp python/lcmaes*.so "$SITE_PKG/"
        log_ok "lcmaes Python バインディング: インストール完了"
        LCMAES_OK=true
      else
        log_warn "lcmaes のビルドは完了しましたが Python バインディングが見つかりません。"
        log_warn "  → Optimization/libs/cmaes.py は手動で対応してください。"
      fi
    ) || {
      log_warn "lcmaes のビルドに失敗しました (非クリティカル)。"
      log_warn "  → optimize_1setup.py / optimize_2setups.py は 'import cma' (pycma) を使うため動作します。"
    }
  fi
fi

# ==============================================================================
#  STEP 3: Python 環境
# ==============================================================================
if $DO_PYTHON; then
log_section "STEP 3: Python 環境セットアップ"

# Python 3.11 を見つける
PYTHON_BIN=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    PY_VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

[[ -z "$PYTHON_BIN" ]] && die "Python 3.10 以上が見つかりません。インストール後に再実行してください。"
log_ok "Python: $($PYTHON_BIN --version)  ($PYTHON_BIN)"

PIP_CMD="$PYTHON_BIN -m pip"

# pip をアップグレード
log_info "pip をアップグレード中..."
$PIP_CMD install --upgrade pip -q

# ── PyTorch (GPU/CPU 自動選択) ────────────────────────────────────────────────
log_info "PyTorch をインストール中..."

if $IS_MACOS && [[ "$ARCH" == "arm64" ]]; then
  # Apple Silicon: MPS バックエンド (Metal) 対応版
  $PIP_CMD install torch torchvision torchaudio -q
  log_ok "PyTorch: Apple Silicon MPS 対応版"
elif $IS_LINUX; then
  # Linux: CUDA が使える場合は CUDA 版、そうでなければ CPU 版
  if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    log_info "NVIDIA GPU 検出: CUDA 対応版をインストール"
    $PIP_CMD install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121 -q
    log_ok "PyTorch: CUDA 12.1 版"
  else
    $PIP_CMD install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu -q
    log_ok "PyTorch: CPU 版"
  fi
else
  $PIP_CMD install torch torchvision torchaudio -q
  log_ok "PyTorch: デフォルト版"
fi

# ── Taichi ────────────────────────────────────────────────────────────────────
log_info "Taichi をインストール中..."
$PIP_CMD install taichi -q
log_ok "Taichi: $(python -c 'import taichi; print(taichi.__version__)' 2>/dev/null || echo 'インストール済み')"

# ── Python パッケージ一覧 ──────────────────────────────────────────────────────
log_info "Python パッケージをインストール中..."

PACKAGES=(
  # 最適化
  "cma"                   # CMA-ES (pycma)
  "gpytorch"              # Gaussian Process
  "joblib"                # モデル保存・並列処理

  # 機械学習
  "scikit-learn"

  # 数値・データ処理
  "numpy"
  "scipy"
  "pandas"

  # 画像処理・コンピュータビジョン
  "opencv-python"
  "Pillow"

  # 可視化
  "matplotlib"
  "seaborn"
  "plotly"

  # ユーティリティ
  "tqdm"
  "psutil"
  "streamlit"             # FlowCurve/param.py で使用
)

FAILED_PACKAGES=()
for pkg in "${PACKAGES[@]}"; do
  if $PIP_CMD install "$pkg" -q; then
    log_ok "${pkg}"
  else
    log_warn "${pkg}: インストール失敗 (スキップ)"
    FAILED_PACKAGES+=("$pkg")
  fi
done

# requirements.txt を生成 (再現性のため)
log_info "requirements.txt を生成中..."
$PIP_CMD freeze > "${PROJECT_ROOT}/requirements.txt"
log_ok "requirements.txt を出力: ${PROJECT_ROOT}/requirements.txt"

fi  # DO_PYTHON

# ==============================================================================
#  STEP 4: C++ コンポーネントのビルド
# ==============================================================================
if $DO_CPP; then
log_section "STEP 4: C++ コンポーネントのビルド"

# CMake バージョン確認
if ! command -v cmake &>/dev/null; then
  die "cmake が見つかりません。STEP 1 が完了しているか確認してください。"
fi
log_ok "CMake: $(cmake --version | head -1)"

NCPU=$(nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

# Eigen3 パスの解決
EIGEN3_HINT=""
if $IS_MACOS; then
  EIGEN3_HINT="-DEIGEN3_INCLUDE_DIR=${HOMEBREW_PREFIX}/include/eigen3"
elif $IS_LINUX; then
  EIGEN3_HINT="-DEIGEN3_INCLUDE_DIR=/usr/include/eigen3"
fi

CPP_BUILD_ERRORS=()

# ── 4a. GLRender3d ─────────────────────────────────────────────────────────
GLRENDER_SRC="${PROJECT_ROOT}/Simulation/GLRender3d"
GLRENDER_BUILD="${GLRENDER_SRC}/build"
log_info "GLRender3d をビルド中..."

if [ ! -d "$GLRENDER_SRC" ]; then
  log_warn "Simulation/GLRender3d が見つかりません。スキップします。"
else
  (
    set -e
    rm -rf "$GLRENDER_BUILD"
    mkdir -p "$GLRENDER_BUILD"
    cd "$GLRENDER_BUILD"

    CMAKE_ARGS=(
      ".."
      "-DCMAKE_BUILD_TYPE=Release"
      $EIGEN3_HINT
    )

    if $IS_MACOS; then
      CMAKE_ARGS+=(
        "-DCMAKE_OSX_ARCHITECTURES=${ARCH}"
        "-DOpenGL_GL_PREFERENCE=LEGACY"
      )
    fi

    cmake "${CMAKE_ARGS[@]}" > /tmp/cmake_gl.log 2>&1 || { tail -10 /tmp/cmake_gl.log; exit 1; }
    make -j"$NCPU" > /tmp/make_gl.log 2>&1    || { tail -10 /tmp/make_gl.log; exit 1; }
    log_ok "GLRender3d: ビルド成功 → ${GLRENDER_BUILD}/GLRender3d"
  ) || {
    log_warn "GLRender3d のビルドに失敗しました。"
    CPP_BUILD_ERRORS+=("GLRender3d")
  }
fi

# ── 4b. cpp_marching_cubes ─────────────────────────────────────────────────
MC_SRC="${PROJECT_ROOT}/Simulation/ParticleSkinner3DTaichi/cpp_marching_cubes"
MC_BUILD="${MC_SRC}/build"
log_info "cpp_marching_cubes をビルド中..."

if [ ! -d "$MC_SRC" ]; then
  log_warn "cpp_marching_cubes が見つかりません。スキップします。"
else
  (
    set -e
    # Eigen5 互換: CMakeLists.txt のバージョン制約を除去
    sed -i.bak 's/find_package( Eigen3 [0-9.]* REQUIRED )/find_package( Eigen3 REQUIRED )/' "$MC_SRC/CMakeLists.txt"

    rm -rf "$MC_BUILD"
    mkdir -p "$MC_BUILD"
    cd "$MC_BUILD"

    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      $EIGEN3_HINT \
      > /tmp/cmake_mc.log 2>&1 || { tail -10 /tmp/cmake_mc.log; exit 1; }
    make -j"$NCPU" > /tmp/make_mc.log 2>&1 || { tail -10 /tmp/make_mc.log; exit 1; }
    log_ok "cpp_marching_cubes: ビルド成功 → ${MC_BUILD}/cpp_marching_cubes"
  ) || {
    log_warn "cpp_marching_cubes のビルドに失敗しました。"
    CPP_BUILD_ERRORS+=("cpp_marching_cubes")
  }
fi

fi  # DO_CPP

# ==============================================================================
#  STEP 5: インポートの動作確認
# ==============================================================================
log_section "STEP 5: インポートチェック"

if $DO_PYTHON; then
  PYTHON_CHECK_CMD="$PYTHON_BIN"

  IMPORT_CHECKS=(
    "taichi"
    "torch"
    "gpytorch"
    "cma"
    "joblib"
    "sklearn"
    "numpy"
    "scipy"
    "pandas"
    "cv2"
    "PIL"
    "matplotlib"
    "seaborn"
    "plotly"
    "tqdm"
    "psutil"
  )

  IMPORT_ERRORS=()
  for mod in "${IMPORT_CHECKS[@]}"; do
    if "$PYTHON_CHECK_CMD" -c "import ${mod}" 2>/dev/null; then
      log_ok "import ${mod}"
    else
      log_warn "import ${mod}  ← インポート失敗"
      IMPORT_ERRORS+=("$mod")
    fi
  done

  # lcmaes は非クリティカル
  if "$PYTHON_CHECK_CMD" -c "import lcmaes" 2>/dev/null; then
    log_ok "import lcmaes"
  else
    log_warn "import lcmaes  ← 非クリティカル (libs/cmaes.py のみ影響)"
  fi
fi

# ==============================================================================
#  完了サマリー
# ==============================================================================
log_section "セットアップ完了"

echo -e "${BOLD}実行方法:${NC}"
echo -e "  ${CYAN}# カメラキャリブレーション${NC}"
echo -e "  cd Calibration && python pipeline.py --calib_img IMG.JPG --target config_00.png"
echo ""
echo -e "  ${CYAN}# MPM シミュレーション${NC}"
echo -e "  cd Simulation && python main.py --eta 4.54 --n 0.999 --sigma_y 1.007 --ref ref_dir"
echo ""
echo -e "  ${CYAN}# パラメータ最適化 (1セットアップ)${NC}"
echo -e "  cd Optimization && python optimize_1setup.py --moe_dir moe_workspace5 \\"
echo -e "      -W1 2.0 -H1 3.5 -dis1 0.409 1.010 1.421 1.617 1.730 1.816 1.890 1.954 \\"
echo -e "      --strategy topk --topk 2"

# エラーサマリー
TOTAL_ERRORS=0

if $DO_PYTHON && [ "${#IMPORT_ERRORS[@]}" -gt 0 ]; then
  echo ""
  log_warn "インポートに失敗したパッケージ:"
  for m in "${IMPORT_ERRORS[@]}"; do echo "    - $m"; done
  TOTAL_ERRORS=$((TOTAL_ERRORS + ${#IMPORT_ERRORS[@]}))
fi

if $DO_CPP && [ "${#CPP_BUILD_ERRORS[@]}" -gt 0 ]; then
  echo ""
  log_warn "ビルドに失敗した C++ コンポーネント:"
  for c in "${CPP_BUILD_ERRORS[@]}"; do echo "    - $c"; done
  TOTAL_ERRORS=$((TOTAL_ERRORS + ${#CPP_BUILD_ERRORS[@]}))
fi

if [ "$TOTAL_ERRORS" -eq 0 ]; then
  echo ""
  echo -e "${GREEN}${BOLD}  すべてのセットアップが完了しました。${NC}"
else
  echo ""
  echo -e "${YELLOW}${BOLD}  セットアップは完了しましたが、${TOTAL_ERRORS} 件の警告があります。${NC}"
  echo -e "${YELLOW}  上記の警告を確認してください。${NC}"
fi

echo ""
