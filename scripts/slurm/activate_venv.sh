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
#     4. torch から GPU が使えるか。使えなければ、このノードを外して同じジョブを自動で投げ直してから止まる
#        （ALLOW_CPU=1 なら警告だけ出して CPU で続ける）
#
#   GPU が無いときの投げ直しに使う環境変数:
#     EXCLUDE_NODES  今までに外したノード（カンマ区切り）。投げ直すたびに、このノードを足して引き継ぐ
#     RESUBMIT_LEFT  あと何回投げ直してよいか（既定 3）。0 なら投げ直さずに止まる
# =====================================================================

_venv="$1"
_job_script="$2"
# Slurm のノード名（--exclude に使う）。ホスト名と違うクラスタもあるので、Slurm が付ける名前を優先する
_node="${SLURMD_NODENAME:-$(hostname -s)}"
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
# デバイスの auto は黙って CPU になり、何倍も時間がかかる（2 日の時間制限に収まらないこともある）。
# そうなる前に、このノードを外して同じジョブを投げ直し、このジョブはエラーで終わる:
#   - 外したノードは環境変数 EXCLUDE_NODES（カンマ区切り）に溜めて引き継ぐので、投げ直すたびに外すノードが増える
#   - 投げ直すのは RESUBMIT_LEFT 回まで（既定 3）。使い切ったら、手で投げ直す方法を表示して止まる
#   - 配列ジョブなら、この番号だけの配列ジョブとして投げ直す（--array=<番号>）。パーティションも同じにする
#   - 環境変数は --export=ALL で全て引き継ぐ（ジョブスクリプトは OUT などを export しておく）
#   - sbatch のコマンドラインで付けたオプション（-c, -t, -J, -w, --dependency など）は引き継がない。
#     投げ直したジョブは、ジョブスクリプトの #SBATCH の既定値で動く
#   - ALLOW_CPU=1 なら投げ直さず、警告だけ出して CPU で続ける
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [[ "${ALLOW_CPU:-0}" == 1 ]]; then
        echo "警告: ${_node} では GPU が使えません。ALLOW_CPU=1 なので CPU で実行します"
    else
        echo "エラー: ${_node} では PyTorch から GPU が使えません（上の nvidia-smi の出力を確認）。" >&2
        _left="${RESUBMIT_LEFT:-3}"
        _exclude="${EXCLUDE_NODES:+${EXCLUDE_NODES},}${_node}"
        if [[ -n "${SLURM_JOB_ID:-}" && "${_left}" -gt 0 ]]; then
            # --export=ALL,EXCLUDE_NODES=a,b とは書けない（Slurm は --export の値をカンマで区切るので、ノードが 2 つ以上で壊れる）。
            # 先に環境変数にしておき、--export=ALL でまとめて引き継ぐ
            export EXCLUDE_NODES="${_exclude}" RESUBMIT_LEFT="$((_left - 1))"
            _resubmit=(sbatch --parsable --exclude="${_exclude}" --export=ALL)
            if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
                _resubmit+=(--array="${SLURM_ARRAY_TASK_ID}")
            fi
            if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
                _resubmit+=(-p "${SLURM_JOB_PARTITION}")
            fi
            _resubmit+=("${_job_script}")
            echo "  このノードを外して投げ直します（残り $((_left - 1)) 回）: ${_resubmit[*]}" >&2
            if _new_job="$("${_resubmit[@]}")"; then
                echo "  投げ直したジョブ: ${_new_job}（外したノード: ${_exclude}）" >&2
            else
                echo "  投げ直しに失敗しました。手で投げ直してください: sbatch --exclude=${_exclude} ... ${_job_script}" >&2
            fi
        else
            echo "  このノードを外して投げ直してください: sbatch --exclude=${_exclude}${SLURM_ARRAY_TASK_ID:+ --array=${SLURM_ARRAY_TASK_ID}}${SLURM_JOB_PARTITION:+ -p ${SLURM_JOB_PARTITION}} ... ${_job_script}" >&2
            echo "  CPU で実行してよい場合は ALLOW_CPU=1 を付けて投げてください" >&2
        fi
        exit 1
    fi
fi
echo "python: $(command -v python)（${_node_ver}）、torch $(python -c 'import torch; print(torch.__version__)')"
