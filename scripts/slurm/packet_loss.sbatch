#!/bin/bash
# =====================================================================
# packet_loss_sweep.py を研究室の Slurm クラスタで実行するジョブスクリプト。
#   配列ジョブで、設定の targets の 1 つ（= 1 モデル）ずつを別々のジョブとして並列に動かす。
#   結果は全ての配列のジョブで 1 つのフォルダ runs/<設定ファイル名>_<配列ジョブ ID>/ に入る:
#     <モデル名>/            配列のジョブ 1 つ分の実行フォルダ（ジョブ同士でファイルを共有しないので、書き込みはぶつからない）
#     results.csv など       全モデル分をまとめたもの。配列のジョブが全て終わると、まとめ用のジョブ
#                            （scripts/slurm/packet_loss_merge.sbatch）が自動で動いて作る。figures/ に図も作る
#   手順の詳しい説明は docs/experiments.md の「Slurm クラスタで動かす」を参照。
#
#   リポジトリのトップで投げる（ログの出力先 runs/slurm/ は先に作っておく。Slurm は作ってくれない）:
#     mkdir -p runs/slurm
#     sbatch scripts/slurm/packet_loss.sbatch                     # 全 4 モデル（配列 0-3）
#     CONFIG=configs/packet_loss_ucf101_quick.yaml sbatch --array=0-1 scripts/slurm/packet_loss.sbatch   # 別の設定（target 2 つ）
#     SCHEMES="F+Sym D+DZ" sbatch scripts/slurm/packet_loss.sbatch  # 方式を絞る
#     sbatch --array=1 scripts/slurm/packet_loss.sbatch           # target 1 番（resnet50）だけ
#     EXTRA="data.kwargs.num_clips=2 seeds=[0]" sbatch --array=1 scripts/slurm/packet_loss.sbatch   # 動作確認
#     OUT=runs/packet_loss_ucf101_12345 sbatch --array=1 scripts/slurm/packet_loss.sbatch   # 時間切れ・失敗したモデルを同じフォルダで再開
#     RESUME=runs/packet_loss_ucf101_resnet50_20261023-120000 sbatch --array=1 scripts/slurm/packet_loss.sbatch  # 以前の形式の実行フォルダを再開
#
#   環境変数（sbatch は投げたシェルの環境変数をジョブに引き継ぐ）:
#     CONFIG   設定ファイル（既定 configs/packet_loss_ucf101.yaml）
#     SCHEMES  実行する方式の name（空白区切り）。python の --scheme に当たる。既定は空で、設定の schemes を全て実行する
#     EXTRA    packet_loss_sweep.py にそのまま渡す追加の引数（key=value の上書きなど。空白区切り）
#     OUT      結果をまとめるフォルダ（既定 runs/<設定ファイル名>_<配列ジョブ ID>。全ての配列のジョブで同じになる）。
#              既にあるフォルダを指定すると、そのモデルのフォルダがあれば続きから再開する（CONFIG / SCHEMES は無視される）
#     NO_MERGE 1 にすると、まとめ用のジョブを投げない（あとで python scripts/merge_results.py --group <OUT> で手動でまとめる）
#     RESUME   以前の形式（runs/<実験名>_<モデル名>_<日時>）の実行フォルダ。指定すると --resume で続きから再開する
#              （CONFIG / SCHEMES / OUT は無視され、まとめ用のジョブも投げない）
#     VENV     有効にする仮想環境（既定 .venv。scripts/slurm/setup_env.sh で作ったもの）
#     DRY_RUN  1 にすると実行するコマンドを表示するだけで終わる（スクリプトの確認用）
#     ALLOW_CPU 1 にすると、GPU が使えないノードでも CPU で実行する（既定では GPU が無ければエラーで止まる）
#
#   targets の番号は python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets で確認できる。
# =====================================================================

#SBATCH -J packet_loss              # ジョブ名（squeue とログのファイル名に出る）
#SBATCH -p GPU                      # GPU のあるノード。GRES は未設定なので --gres は付けない
#SBATCH -N 1                        # ノード数
#SBATCH -c 12                       # CPU コア数。指定しないと 1 コアになり、符号化（CPU）が極端に遅くなる
#                                   tail が CPU（ALLOW_CPU=1 など）のときは、この数のワーカープロセスで並列に処理する
#                                   （ワーカー 1 つあたり約 1GB のメモリが要る。数を変えたいときは EXTRA="--workers 8"）
#SBATCH -t 2-00:00:00               # 時間制限（GPU パーティションの上限 2 日）。切れたら RESUME で再開する
#SBATCH --array=0-3                 # 配列の番号 = 設定の targets の番号
#SBATCH -o runs/slurm/%x_%A_%a.out  # 標準出力（%A: 配列ジョブ ID, %a: 配列の番号）
#SBATCH -e runs/slurm/%x_%A_%a.err  # 標準エラー（進捗バーもここに出る）

set -euo pipefail

# sbatch を実行したディレクトリ（リポジトリのトップ）で動かす。Slurm 外で直接 bash で実行した場合はその場所
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-configs/packet_loss_ucf101.yaml}"
SCHEMES="${SCHEMES:-}"
EXTRA="${EXTRA:-}"
RESUME="${RESUME:-}"
VENV="${VENV:-.venv}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-local}"
OUT="${OUT:-runs/$(basename "$CONFIG" .yaml)_${ARRAY_JOB}}"
NO_MERGE="${NO_MERGE:-0}"

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
    cmd+=(--config "$CONFIG" --target "$TASK" --group-dir "$OUT")
    for s in ${scheme_names[@]+"${scheme_names[@]}"}; do
        cmd+=(--scheme "$s")
    done
fi
# ${a[@]+"${a[@]}"} は空の配列でも set -u で止まらないための書き方（bash 4.3 以前）
cmd+=(${extra_args[@]+"${extra_args[@]}"})

echo "=== $(date '+%F %T') ジョブ ${SLURM_JOB_ID:-なし}（配列 ${TASK}）: $(hostname)"
echo "コマンド: ${cmd[*]}"

# まとめ用のジョブ: 配列の中で番号が一番小さいジョブだけが、配列のジョブが全て終わってから動くジョブを 1 つ投げる
# （afterany なので、失敗・時間切れのモデルがあっても、終わった分だけでまとめる）
merge_cmd=()
if [[ -z "$RESUME" && "$NO_MERGE" != 1 && -n "${SLURM_ARRAY_JOB_ID:-}" && "$TASK" == "${SLURM_ARRAY_TASK_MIN:-$TASK}" ]]; then
    merge_cmd=(sbatch --parsable --dependency="afterany:${SLURM_ARRAY_JOB_ID}" --export="ALL,OUT=${OUT}")
    if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
        merge_cmd+=(-p "$SLURM_JOB_PARTITION")
    fi
    merge_cmd+=(scripts/slurm/packet_loss_merge.sbatch)
    echo "まとめ用のジョブ: ${merge_cmd[*]}"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
if [[ ${#merge_cmd[@]} -gt 0 ]]; then
    if merge_job="$("${merge_cmd[@]}")"; then
        echo "まとめ用のジョブ ${merge_job} を投げました（配列ジョブ ${SLURM_ARRAY_JOB_ID} が全て終わると ${OUT} にまとめる）"
    else
        echo "警告: まとめ用のジョブを投げられませんでした。全て終わったら手動でまとめてください:" >&2
        echo "  python scripts/merge_results.py --group ${OUT} && python scripts/packet_loss_plot.py ${OUT}" >&2
    fi
fi

# 仮想環境を有効にし、torch と GPU が使えるかを確かめる（だめなら原因を表示して止まる）
source scripts/slurm/activate_venv.sh "$VENV" scripts/slurm/packet_loss.sbatch

"${cmd[@]}"
echo "=== $(date '+%F %T') 終了"
