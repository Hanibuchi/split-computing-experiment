#!/bin/bash
# =====================================================================
# 研究室の Slurm クラスタで使う仮想環境（.venv）を uv で作る。ログインノードで、リポジトリのトップで実行する。
#   bash scripts/slurm/setup_env.sh
#
#   ノードごとに OS が違い、/usr/bin/python3 のバージョンも違う（例: ログインノード 3.10、計算ノード 3.12）。
#   システムの python で作った venv は、そのバージョン用にパッケージが入るので、別のバージョンのノードでは
#   torch などが見つからなくなる。そこで uv で python 自体をホーム（~/.local/share/uv/python/、全ノードで共有）に入れ、
#   その python で venv を作る。こうすると、どのノードでも同じ python・同じパッケージで動く。
#
#   - uv が無ければ ~/.local/bin に入れる
#   - .venv が既にあれば作り直す（パッケージも入れ直す。数分かかる）
#   - 何度実行してもよい。依存関係（pyproject.toml）を変えたときも、これを実行し直せばよい
#
#   環境変数:
#     PYTHON_VERSION  使う python のバージョン（既定 3.12）
#     VENV            作る仮想環境（既定 .venv）
#     EXTRAS          pyproject.toml の optional-dependencies（既定 video,analysis）
# =====================================================================

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv}"
EXTRAS="${EXTRAS:-video,analysis}"

if ! command -v uv > /dev/null; then
    echo "=== uv をインストールします（~/.local/bin）"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "=== $(uv --version)"

# python をホームに入れる（入っていれば何もしない）。--managed-python で、システムの python は使わない
uv python install "$PYTHON_VERSION"
uv venv --managed-python --python "$PYTHON_VERSION" --clear "$VENV"
uv pip install --python "$VENV/bin/python" -e ".[$EXTRAS]"

echo "=== 確認"
"$VENV/bin/python" -c "import sys, torch; print(sys.executable, sys.version.split()[0], 'torch', torch.__version__)"
echo "python の実体: $(readlink -f "$VENV/bin/python")"
echo "完了。使うときは: source $VENV/bin/activate"
