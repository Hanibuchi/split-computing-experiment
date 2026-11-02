#!/bin/bash
# =====================================================================
# packet_loss.sbatch の配列ジョブが全て終わったあとに、結果を 1 つにまとめて図を作るジョブスクリプト。
#   packet_loss.sbatch が自動で投げる（--dependency=afterany:<配列ジョブ ID>）ので、ふつうは手で投げなくてよい。
#   OUT（runs/<設定ファイル名>_<配列ジョブ ID>）の下のモデルごとのフォルダをまとめ、OUT に
#   results.csv / gop_curve.csv（と .jsonl, sources.txt）、OUT/figures/ に図と summary.md を作る。
#
#   手で投げ直す場合（あとから再開したモデルの分を足すときなど）:
#     OUT=runs/packet_loss_ucf101_12345 sbatch scripts/slurm/packet_loss_merge.sbatch
#   ログインノードで直接実行してもよい（軽い）:
#     python scripts/merge_results.py --group runs/packet_loss_ucf101_12345 && python scripts/packet_loss_plot.py runs/packet_loss_ucf101_12345
#
#   環境変数:
#     OUT   まとめるフォルダ（必須）
#     VENV  有効にする仮想環境（既定 .venv）
# =====================================================================

#SBATCH -J packet_loss_merge        # ジョブ名
#SBATCH -p GPU                      # packet_loss.sbatch からはそのジョブと同じパーティションで投げる。GPU は使わない
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 01:00:00
#SBATCH -o runs/slurm/%x_%j.out
#SBATCH -e runs/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUT="${OUT:?OUT（まとめるフォルダ）を指定してください}"
VENV="${VENV:-.venv}"

echo "=== $(date '+%F %T') まとめ: ${OUT}（ジョブ ${SLURM_JOB_ID:-なし}: $(hostname)）"
# まとめと作図は CPU だけで足りるので、GPU が使えないノードでも止めない
export ALLOW_CPU=1
source scripts/slurm/activate_venv.sh "$VENV" scripts/slurm/packet_loss_merge.sbatch

python scripts/merge_results.py --group "$OUT"
python scripts/packet_loss_plot.py "$OUT"
echo "=== $(date '+%F %T') 終了: ${OUT}/results.csv, ${OUT}/figures/"
