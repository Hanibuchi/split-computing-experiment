"""モデルを任意の位置で head（端末側）と tail（サーバ側）に分割する。

仕組み
------
``torch.fx`` でモデルの forward を「計算ノードの列（グラフ）」に変換し、
その列の k 番目のノードの直後で切る。

    ノード列:  x → conv1 → bn1 → relu → ... → fc → output
                              ↑ ここで切る（分割点 = "bn1"）

ResNet のような skip connection があるモデルでは、切った位置をまたいで
後ろで使われる値が複数あることがある。そこで

    「分割点以前に計算され、かつ分割点より後で使われる値」すべて

を head の出力（= 送信するテンソル群）とし、tail はそれらを入力として受け取る。
これによりどんな構造のモデルでも ``tail(*head(x)) == model(x)`` が成り立つ。

- 転送テンソルが 1 つだけの分割点を「clean な分割点」と呼ぶ。
  学習型ボトルネックを挿入できるのは clean な分割点のみ。
- ``trace_depth`` を指定すると、その深さより深いモジュールを展開せず 1 ノードとして扱う。
  例: ResNet で trace_depth=1 → conv1, bn1, relu, maxpool, layer1, ..., fc という粗い分割点。

使い方::

    splitter = Splitter(model, trace_depth=1)
    for p in splitter.split_points():           # 分割点の一覧
        print(p.index, p.name, p.num_tensors)
    head, tail = splitter.split("layer2")
    z = head(x)          # tuple[Tensor, ...]（送信するテンソル群）
    y = tail(*z)         # 元のモデルの出力と一致

head の出力は常に tuple、tail は tuple の要素を位置引数で受け取る。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch.fx as fx
import torch.nn as nn

#: 分割点名のエイリアス: "input" = 何も計算せず入力を送る（全てサーバで処理）
INPUT_ALIAS = "input"
#: 分割点名のエイリアス: "output" = 全て端末で計算し、出力を送る（全て端末で処理）
OUTPUT_ALIAS = "output"


@dataclass
class SplitPoint:
    """分割点の情報。

    Attributes:
        index: ノード列の中での位置（このノードの直後で切る）。
        name: ノード名（split() に渡す名前）。"layer1_0_relu" のような形式。
        op: fx のノード種別（call_module / call_function / call_method / placeholder ...）。
        target: 呼び出し先（モジュール名や関数名）。
        num_tensors: この位置で切ったときに送る値の個数。
        crossing: 送る値を作るノード名のリスト。
    """

    index: int
    name: str
    op: str
    target: str
    num_tensors: int
    crossing: list[str]

    @property
    def is_clean(self) -> bool:
        """送る値が 1 つだけか（skip connection の途中で切っていないか）。"""
        return self.num_tensors == 1


class _DepthLimitedTracer(fx.Tracer):
    """指定した深さより深いモジュールを展開しない Tracer。"""

    def __init__(self, max_depth: int | None):
        super().__init__()
        self.max_depth = max_depth

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if self.max_depth is not None and module_qualified_name:
            if module_qualified_name.count(".") + 1 >= self.max_depth:
                return True
        return super().is_leaf_module(m, module_qualified_name)


def trace(model: nn.Module, trace_depth: int | None = None) -> fx.GraphModule:
    """モデルを torch.fx でグラフ化する。"""
    tracer = _DepthLimitedTracer(trace_depth)
    graph = tracer.trace(model)
    name = model.__class__.__name__
    return fx.GraphModule(tracer.root, graph, name)


class Splitter:
    """1 つのモデルをいろいろな分割点で切るためのクラス（グラフ化は最初の 1 回だけ行う）。

    注意: 生成される head / tail は元のモデルとパラメータを共有する。
    学習で重みを変える場合は ``copy.deepcopy(model)`` を渡すこと。

    Args:
        model: 分割したいモデル。
        trace_depth: グラフ化の展開深さ（モジュール docstring 参照）。None で全て展開。
    """

    def __init__(self, model: nn.Module, trace_depth: int | None = None):
        self.gm = trace(model, trace_depth)
        self.nodes: list[fx.Node] = list(self.gm.graph.nodes)
        self._index = {n.name: i for i, n in enumerate(self.nodes)}
        self._num_inputs = sum(1 for n in self.nodes if n.op == "placeholder")
        self._points = self._compute_points()

    # ------------------------------------------------------------------ 分割点の列挙

    def _compute_points(self) -> list[SplitPoint]:
        """全ての分割候補について「切ったときに送る値」を求める。

        各ノードについて「定義された位置」と「最後に使われる位置」を求めておけば、
        位置 k で切ったときに送る値 = 定義位置 <= k < 最終使用位置 のノード、となる。
        """
        last_use: dict[fx.Node, int] = {}
        for i, n in enumerate(self.nodes):
            for inp in n.all_input_nodes:
                last_use[inp] = i
        points = []
        # 入力（placeholder）の後ろから、output ノードの手前までが分割候補
        for k in range(self._num_inputs - 1, len(self.nodes) - 1):
            node = self.nodes[k]
            if node.op == "get_attr":
                continue
            crossing = [
                n
                for n in self.nodes[: k + 1]
                # get_attr（パラメータや定数の参照）は送らずに tail 側で参照し直す
                if n.op != "get_attr" and last_use.get(n, -1) > k
            ]
            points.append(
                SplitPoint(
                    index=k,
                    name=node.name,
                    op=node.op,
                    target=str(node.target),
                    num_tensors=len(crossing),
                    crossing=[n.name for n in crossing],
                )
            )
        return points

    def split_points(self, kind: str = "all") -> list[SplitPoint]:
        """分割点の一覧を返す。

        Args:
            kind: "all"（全て）または "clean"（送る値が 1 つだけのもの）。
        """
        if kind == "all":
            return list(self._points)
        if kind == "clean":
            return [p for p in self._points if p.is_clean]
        raise ValueError(f"kind は 'all' か 'clean' を指定してください: {kind!r}")

    def get_point(self, name: str) -> SplitPoint:
        """分割点名（またはエイリアス "input" / "output"）から SplitPoint を返す。"""
        if name == INPUT_ALIAS:
            return self._points[0]
        if name == OUTPUT_ALIAS:
            return self._points[-1]
        for p in self._points:
            if p.name == name:
                return p
        raise KeyError(
            f"分割点 {name!r} が見つかりません。scripts/list_splits.py で候補を確認してください"
            f"（例: {[p.name for p in self.split_points('clean')[:8]]} ...）"
        )

    def resolve(self, spec) -> list[SplitPoint]:
        """設定ファイルの split_points 指定（"clean" / "all" / 名前のリスト）を SplitPoint のリストに変換する。"""
        if isinstance(spec, str):
            if spec in ("all", "clean"):
                return self.split_points(spec)
            spec = [spec]
        return [self.get_point(s) for s in spec]

    # ------------------------------------------------------------------ 分割

    def split(self, point: str | SplitPoint) -> tuple[fx.GraphModule, fx.GraphModule]:
        """指定した分割点で (head, tail) を作る。

        Returns:
            head: 入力を受け取り、送信する値の tuple を返すモジュール。
            tail: 送信された値を位置引数で受け取り、元のモデルと同じ出力を返すモジュール。
        """
        if isinstance(point, str):
            point = self.get_point(point)
        k = point.index
        crossing = [self.nodes[self._index[name]] for name in point.crossing]

        # --- head: 先頭から k 番目のノードまでをコピーし、送る値を tuple で返す
        head_graph = fx.Graph()
        env: dict[fx.Node, fx.Node] = {}
        for n in self.nodes[: k + 1]:
            env[n] = head_graph.node_copy(n, lambda a: env[a])
        head_graph.output(tuple(env[c] for c in crossing))

        # --- tail: 送られてきた値を placeholder として受け取り、残りのノードをコピー
        tail_graph = fx.Graph()
        tenv: dict[fx.Node, fx.Node] = {}
        for i, c in enumerate(crossing):
            tenv[c] = tail_graph.placeholder(f"z{i}_{c.name}")

        def lookup(a: fx.Node) -> fx.Node:
            if a in tenv:
                return tenv[a]
            if a.op == "get_attr":  # head 側で定義されたパラメータ参照は tail 側で作り直す
                tenv[a] = tail_graph.get_attr(a.target)
                return tenv[a]
            raise RuntimeError(f"内部エラー: ノード {a.name} が tail 側で解決できません")

        for n in self.nodes[k + 1 :]:
            tenv[n] = tail_graph.node_copy(n, lookup)

        # GraphModule(root, graph) は graph 内で参照されるサブモジュール・パラメータだけを持つ
        head = fx.GraphModule(self.gm, head_graph, f"Head_{point.name}")
        tail = fx.GraphModule(self.gm, tail_graph, f"Tail_{point.name}")
        for m in (head, tail):
            # 使われなくなった計算（例: head 側で計算したが tail でしか使わない定数参照）を削除
            m.graph.eliminate_dead_code()
            m.delete_all_unused_submodules()
            m.recompile()
            m.train(self.gm.training)
        return head, tail


# ---------------------------------------------------------------------------
# fx でグラフ化できないモデル向けの簡易分割
# ---------------------------------------------------------------------------


class _SeqHead(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return (self.layers(x),)


def split_sequential(model: nn.Module, index: int) -> tuple[nn.Module, nn.Module]:
    """モデルの直下の子モジュールを順番に適用するだけ、とみなして分割する簡易版。

    torch.fx でグラフ化できない（forward 内で入力に依存する if 文がある等）モデル向け。
    forward が子モジュールを順に呼ぶだけの構造でないと正しく動かないので注意。

    Args:
        index: 先頭から index 個の子モジュールを head に入れる。
    """
    children = list(model.children())
    if not 0 <= index <= len(children):
        raise IndexError(f"index は 0〜{len(children)} の範囲で指定してください")
    warnings.warn("split_sequential は forward が子モジュールの順次適用である前提です。結果を必ず検証してください")
    return _SeqHead(children[:index]), nn.Sequential(*children[index:])
