#!/bin/bash
# =====================================================================
# verify_shiba.py（Shiba et al. の再現検証）を研究室の Slurm クラスタで実行するジョブスクリプト。
#   配列ジョブで、論文の 12 条件（モデル × 分割点）の 1 つずつを別々のジョブとして並列に計算する。
#   全てのジョブが同じ実行フォルダの raw/ に結果を書くので、全部終わったら --report-only で報告書を作る。
#   手順の詳しい説明は docs/experiments.md のレシピ 6 を参照。
#
#   リポジトリのトップで投げる（ログの出力先 runs/slurm/ は先に作っておく。Slurm は作ってくれない）:
#     mkdir -p runs/slurm
#     sbatch scripts/slurm/verify_shiba.sbatch                       # 12 条件全て（配列 0-11）
#     EXTRA="--num-clips 2 --quick" sbatch --array=4 scripts/slurm/verify_shiba.sbatch   # 動作確認（RN-L2 だけ）
#     OUT=runs/verify_shiba_12345 sbatch --array=3,7 scripts/slurm/verify_shiba.sbatch   # 足りない条件だけ同じフォルダに追加
#   終わったら（ログインノードで）:
#     python scripts/verify_shiba.py --report-only runs/verify_shiba_<配列ジョブ ID>
#
#   環境変数（sbatch は投げたシェルの環境変数をジョブに引き継ぐ）:
#     OUT       結果の保存先（既定 runs/verify_shiba_<配列ジョブ ID>。全ての配列のジョブで同じフォルダになる）
#     EXTRA     verify_shiba.py にそのまま渡す追加の引数（空白区切り。例: "--quick" / "--num-clips 2"）
#     VENV      有効にする仮想環境（既定 .venv）
#     DRY_RUN   1 にすると実行するコマンドを表示するだけで終わる（スクリプトの確認用）
#     ALLOW_CPU 1 にすると、GPU が使えないノードでも CPU で実行する（既定では GPU が無ければエラーで止まる）
#
#   配列の番号と条件: 0-2 EfficientNetV2-S S3/S4/S6, 3-5 ResNet-50 L1/L2/L3, 6-8 VGG-16 9/16/23,
#   9-11 MobileNetV2 S4/S14/S17（src/splitcomp/temporal/paper_values.py の TARGETS の順）
# =====================================================================

#SBATCH -J verify_shiba             # ジョブ名（squeue とログのファイル名に出る）
#SBATCH -p GPU                      # GPU のあるノード。GRES は未設定なので --gres は付けない
#SBATCH -N 1                        # ノード数
#SBATCH -c 8                        # CPU コア数（量子化・エントロピーは CPU で行う）
#SBATCH -t 2-00:00:00               # 時間制限（GPU パーティションの上限 2 日）。切れたら同じ OUT・番号で投げ直すと続きから
#SBATCH --array=0-11                # 配列の番号 = 論文の条件の番号
#SBATCH -o runs/slurm/%x_%A_%a.out  # 標準出力（%A: 配列ジョブ ID, %a: 配列の番号）
#SBATCH -e runs/slurm/%x_%A_%a.err  # 標準エラー

set -euo pipefail

# sbatch を実行したディレクトリ（リポジトリのトップ）で動かす。Slurm 外で直接 bash で実行した場合はその場所
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
OUT="${OUT:-runs/verify_shiba_${SLURM_ARRAY_JOB_ID:-local}}"
EXTRA="${EXTRA:-}"
VENV="${VENV:-.venv}"

set -f
read -ra extra_args <<< "$EXTRA"
set +f

# 報告書は全ての条件がそろってから 1 回だけ作る（ジョブ同士で report/ への書き込みが競合しないように --no-report）
cmd=(python scripts/verify_shiba.py --target "$TASK" --out "$OUT" --no-report)
cmd+=(${extra_args[@]+"${extra_args[@]}"})

echo "=== $(date '+%F %T') ジョブ ${SLURM_JOB_ID:-なし}（配列 ${TASK}）: $(hostname)"
echo "コマンド: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

# PyTorch から GPU が使えるか確かめる（使えないと --device auto は黙って CPU になり、何倍も時間がかかる）
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    if [[ "${ALLOW_CPU:-0}" == 1 ]]; then
        echo "警告: $(hostname -s) では GPU が使えません。ALLOW_CPU=1 なので CPU で実行します"
    else
        echo "エラー: $(hostname -s) では PyTorch から GPU が使えません（上の nvidia-smi の出力を確認）。" >&2
        echo "  このノードを外して投げ直してください: sbatch --exclude=$(hostname -s) ... scripts/slurm/verify_shiba.sbatch" >&2
        echo "  CPU で実行してよい場合は ALLOW_CPU=1 を付けて投げてください" >&2
        exit 1
    fi
fi

"${cmd[@]}"
echo "=== $(date '+%F %T') 終了。全ての条件が終わったら: python scripts/verify_shiba.py --report-only $OUT"
