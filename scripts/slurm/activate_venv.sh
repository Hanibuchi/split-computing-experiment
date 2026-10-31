#!/bin/bash
# =====================================================================
# ジョブスクリプト（*.sbatch）から source して使う: 仮想環境を有効にし、このノードで実行できるかを確かめる。
#   source scripts/slurm/activate_venv.sh <仮想環境> <ジョブスクリプトのパス>
#
#   次の順に確かめ、だめなら原因と直し方を表示して止まる（exit 1）:
#     1. 仮想環境があるか
#     2. 仮想環境の python がこのノードで起動するか、作ったときと同じバージョンか
#        （システムの python で作った venv は、python のバージョンが違うノードでは使えない）
#     3. torch が import できるか
#     4. torch から GPU が使えるか（ALLOW_CPU=1 なら警告だけ出して CPU で続ける）
# =====================================================================

_venv="$1"
_job_script="$2"
_node="$(hostname -s)"
_setup_hint="ログインノードで bash scripts/slurm/setup_env.sh を実行して、仮想環境を作り直してください"

if [[ ! -f "${_venv}/bin/activate" ]]; then
    echo "エラー: 仮想環境 ${_venv} がありません。${_setup_hint}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${_venv}/bin/activate"

# pyvenv.cfg には作ったときの python のバージョンが書いてある（venv は "version = "、uv は "version_info = "）
_venv_ver="$(sed -n 's/^version\(_info\)\{0,1\} *= *\([0-9]*\.[0-9]*\).*/\2/p' "${_venv}/pyvenv.cfg" | head -n 1)"
_node_ver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2> /dev/null || true)"
if [[ -z "${_node_ver}" ]]; then
    echo "エラー: ${_node} では仮想環境の python（$(readlink -f "${_venv}/bin/python")）が起動できません。" >&2
    echo "  ${_setup_hint}" >&2
    exit 1
fi
if [[ -n "${_venv_ver}" && "${_venv_ver}" != "${_node_ver}" ]]; then
    echo "エラー: 仮想環境 ${_venv} は python ${_venv_ver} 用ですが、${_node} では python ${_node_ver} が起動しました。" >&2
    echo "  システムの python（/usr/bin/python3）で作った venv は、python のバージョンが違うノードでは使えません。" >&2
    echo "  ${_setup_hint}" >&2
    exit 1
fi

if ! python -c "import torch" 2> /dev/null; then
    echo "エラー: 仮想環境 ${_venv}（python ${_node_ver}）に torch が入っていません。${_setup_hint}" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

# PyTorch から GPU が使えるか確かめる。ドライバの不整合（"Driver/library version mismatch"）などで使えないと、
# デバイスの auto は黙って CPU になり、何倍も時間がかかる。そうなる前にエラーで止める
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [[ "${ALLOW_CPU:-0}" == 1 ]]; then
        echo "警告: ${_node} では GPU が使えません。ALLOW_CPU=1 なので CPU で実行します"
    else
        echo "エラー: ${_node} では PyTorch から GPU が使えません（上の nvidia-smi の出力を確認）。" >&2
        echo "  このノードを外して投げ直してください: sbatch --exclude=${_node} ... ${_job_script}" >&2
        echo "  CPU で実行してよい場合は ALLOW_CPU=1 を付けて投げてください" >&2
        exit 1
    fi
fi
echo "python: $(command -v python)（${_node_ver}）、torch $(python -c 'import torch; print(torch.__version__)')"
