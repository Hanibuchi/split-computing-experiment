#!/bin/bash
# =====================================================================
# packet_loss_sweep.py を研究室の Slurm クラスタ（hpcgate）で実行するジョブスクリプト。
#   配列ジョブで、設定の targets の 1 つ（= 1 モデル）ずつを別々のジョブとして並列に動かす。
#   結果は runs/<実験名>_<モデル名>_<日時>/ に出るので、全部終わったら scripts/merge_results.py でまとめる。
#   手順の詳しい説明は docs/experiments.md の「Slurm クラスタで動かす」を参照。
#
#   リポジトリのトップで投げる（ログの出力先 runs/slurm/ は先に作っておく。Slurm は作ってくれない）:
#     mkdir -p runs/slurm
#     sbatch scripts/slurm/packet_loss.sbatch                     # 全 4 モデル（配列 0-3）
#     sbatch --array=1 scripts/slurm/packet_loss.sbatch           # target 1 番（resnet50）だけ
#     EXTRA="data.kwargs.num_clips=2 seeds=[0]" sbatch --array=1 scripts/slurm/packet_loss.sbatch   # 動作確認
#     RESUME=runs/packet_loss_ucf101_resnet50_20261023-120000 sbatch --array=1 scripts/slurm/packet_loss.sbatch  # 再開
#
#   環境変数（sbatch は投げたシェルの環境変数をジョブに引き継ぐ）:
#     CONFIG   設定ファイル（既定 configs/packet_loss_ucf101.yaml）
#     SCHEMES  実行する方式の name（空白区切り。既定 "F+Sym D+Sym D+DZ"。空にすると設定の全方式）
#     EXTRA    packet_loss_sweep.py にそのまま渡す追加の引数（key=value の上書きなど。空白区切り）
#     RESUME   途中で止まった実行フォルダ。指定すると --resume で続きから再開する（CONFIG / SCHEMES は無視される）
#     VENV     有効にする仮想環境（既定 .venv）
#     DRY_RUN  1 にすると実行するコマンドを表示するだけで終わる（スクリプトの確認用）
#     ALLOW_CPU 1 にすると、GPU が使えないノードでも CPU で実行する（既定では GPU が無ければエラーで止まる）
#
#   targets の番号は python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets で確認できる。
# =====================================================================

#SBATCH -J packet_loss              # ジョブ名（squeue とログのファイル名に出る）
#SBATCH -p GPU                      # GPU のあるノード（glockenspiel, marimba, xylophone）。GRES は未設定なので --gres は付けない
#SBATCH -N 1                        # ノード数
#SBATCH -c 12                       # CPU コア数。指定しないと 1 コアになり、符号化（CPU）が極端に遅くなる
#SBATCH -t 2-00:00:00               # 時間制限（GPU パーティションの上限 2 日）。切れたら RESUME で再開する
#SBATCH --array=0-3                 # 配列の番号 = 設定の targets の番号
#SBATCH -o runs/slurm/%x_%A_%a.out  # 標準出力（%A: 配列ジョブ ID, %a: 配列の番号）
#SBATCH -e runs/slurm/%x_%A_%a.err  # 標準エラー（進捗バーもここに出る）

set -euo pipefail

# sbatch を実行したディレクトリ（リポジトリのトップ）で動かす。Slurm 外で直接 bash で実行した場合はその場所
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-configs/packet_loss_ucf101.yaml}"
SCHEMES="${SCHEMES-F+Sym D+Sym D+DZ}"   # :- ではなく - なので、SCHEMES= と空にすると全方式
EXTRA="${EXTRA:-}"
RESUME="${RESUME:-}"
VENV="${VENV:-.venv}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

# 空白で区切って配列にする。set -f で seeds=[0] などがファイル名として展開されないようにする
set -f
read -ra extra_args <<< "$EXTRA"
read -ra scheme_names <<< "$SCHEMES"
set +f

cmd=(python scripts/packet_loss_sweep.py)
if [[ -n "$RESUME" ]]; then
    # 再開: 設定は実行フォルダの config.yaml から読まれる（上書きできるのは devices.* だけ）
    cmd+=(--resume "$RESUME")
else
    cmd+=(--config "$CONFIG" --target "$TASK")
    for s in ${scheme_names[@]+"${scheme_names[@]}"}; do
        cmd+=(--scheme "$s")
    done
fi
# ${a[@]+"${a[@]}"} は空の配列でも set -u で止まらないための書き方（bash 4.3 以前）
cmd+=(${extra_args[@]+"${extra_args[@]}"})

echo "=== $(date '+%F %T') ジョブ ${SLURM_JOB_ID:-なし}（配列 ${TASK}）: $(hostname)"
echo "コマンド: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

# PyTorch から GPU が使えるか確かめる。ドライバの不整合（"Driver/library version mismatch"）などで使えないと、
# devices: auto は黙って CPU になり、何倍も時間がかかる。そうなる前にエラーで止める
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [[ "${ALLOW_CPU:-0}" == 1 ]]; then
        echo "警告: $(hostname -s) では GPU が使えません。ALLOW_CPU=1 なので CPU で実行します"
    else
        echo "エラー: $(hostname -s) では PyTorch から GPU が使えません（上の nvidia-smi の出力を確認）。" >&2
        echo "  このノードを外して投げ直してください: sbatch --exclude=$(hostname -s) ... scripts/slurm/packet_loss.sbatch" >&2
        echo "  CPU で実行してよい場合は ALLOW_CPU=1 を付けて投げてください" >&2
        exit 1
    fi
fi

"${cmd[@]}"
echo "=== $(date '+%F %T') 終了"
