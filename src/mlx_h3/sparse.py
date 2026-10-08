"""Block-sparse attention over Veda tiles: compute 10% of the attention, keep the camera.

Attention is quadratic and H3 packs text, audio and video into ONE jointly attended
sequence, so a step costs the square of everything the shot needs to stay coherent.
Veda (Han et al., ICML 2026) puts a small learned predictor beside the trunk: every
128-token tile of the video grid is pooled into a feature, each head scores
(query tile, key tile) pairs with its own projection, and only the top-scoring tiles
are computed -- per head, per layer, per step, from the activations of that step.

Three rules make this survivable on a model whose failure mode is a wandering camera:

*  **Global rows and columns stay dense.** Text, audio, keyframe anchors and
   references are never sparsified: they are chunked into tiles in sequence order,
   every query sees all of them, and every global query sees everything. The
   video -> video quadrant is the only sparse part, and that quadrant is exactly
   where a band mask loses the long-range geometric context (see the post-mortem in
   mmcat `archive/sparse_attn_AB_2026-10-02.md`, which is what re-opened this).
*  **The budget is fixed per query tile.** Equal-kernel-cost budgets, with the
   fractional part spread by Bresenham, so every query tile attends the same number
   of key tiles -- which is what lets the sparse work be a *dense* kernel.
*  **A tile always keeps itself.** The diagonal is forced in and consumes budget.

Why gather instead of a mask: `mx.fast.scaled_dot_product_attention` applies a mask
AFTER the Q@K^T tile matmul; only its built-in causal mode shortens the key loop.
Handing it a 90%-sparse block mask buys correctness at 0.97x of dense. Because the
budget IS uniform here, the kept key tiles gather into a regular
`[batch, heads, budget, dim]` block and one batched dense call sees a key loop that
is genuinely `budget` long. No custom Metal kernel, no third-party extension: the
speed comes from the shape of the problem.

Nothing here refuses a shape. A canvas the predictor was never searched on still gets
a plan -- the nearest trained one, or a uniform least-padding tile shape -- with a
note saying so, and the render goes ahead. Sparse is a lever the operator pulls; the
operator decides whether the picture survived it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np

#: Rows per tile. Fixed by the predictor's weights, not a tuning knob.
TILE = 128

#: Chunk sizes of the gather pass. The gathered key block is
#: `q_chunk x head_chunk x keys x dim`, so these two numbers bound peak memory.
HEAD_CHUNK = 8
Q_CHUNK = 32

#: Aspects the released predictor carries tile plans for.
ASPECTS = {"16x9": 16 / 9, "9x16": 9 / 16, "4x3": 4 / 3, "3x4": 3 / 4, "1x1": 1.0}

_SCALE = ".__scale"


# --- tile shapes and plans ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TileShape:
    """A (t, h, w) box holding exactly TILE tokens."""

    t: int
    h: int
    w: int

    def __post_init__(self) -> None:
        if self.t * self.h * self.w != TILE:
            raise ValueError(f"tile {self} holds {self.t * self.h * self.w} tokens, want {TILE}")

    def __str__(self) -> str:
        return f"{self.t}x{self.h}x{self.w}"

    @classmethod
    def parse(cls, text: str) -> "TileShape":
        t, h, w = (int(value) for value in text.split("x"))
        return cls(t, h, w)

    def padded(self, grid: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(-(-g // s) * s for g, s in zip(grid, (self.t, self.h, self.w)))

    def num_tiles(self, grid: tuple[int, int, int]) -> int:
        tp, hp, wp = self.padded(grid)
        return tp * hp * wp // TILE

    def transposed(self) -> "TileShape":
        return TileShape(self.t, self.w, self.h)


def all_shapes() -> list[TileShape]:
    """Every power-of-two triple with product TILE."""
    shapes = []
    for i in range(TILE.bit_length()):
        for j in range(TILE.bit_length() - i):
            shapes.append(TileShape(2**i, 2**j, TILE >> (i + j)))
    return sorted(shapes, key=lambda s: (s.t, s.h, s.w))


def candidate_shapes(grid: tuple[int, int, int]) -> list[TileShape]:
    """Shapes whose extents fit inside `grid`; never empty."""
    return [s for s in all_shapes() if s.t <= grid[0] and s.h <= grid[1] and s.w <= grid[2]] or all_shapes()


def least_padding_shape(grid: tuple[int, int, int]) -> TileShape:
    """Least padded, ties broken toward the most cubic shape."""
    return min(
        candidate_shapes(grid),
        key=lambda s: (
            s.num_tiles(grid),
            max(s.t, s.h, s.w) / min(s.t, s.h, s.w),
            (s.t, s.h, s.w),
        ),
    )


def plan_padding(plan: "TilePlan", grid: tuple[int, int, int]) -> float:
    """Worst-case wasted fraction of the tile grid over a live grid, per shape.

    A plan's shapes were searched on its own grid; on any other grid they round the
    extents up, so the padded tile count overshoots the token count. Reported as the
    worst shape, because that shape is what a head will actually pay.
    """
    tokens = math.prod(grid)
    return max(shape.num_tiles(grid) * TILE / tokens - 1.0 for shape in set(plan.shapes))


def aspect_key(width: int, height: int) -> str:
    """The trained aspect the live canvas is closest to, compared in log space."""
    ratio = width / height
    return min(ASPECTS, key=lambda name: abs(math.log(ASPECTS[name] / ratio)))


@dataclasses.dataclass(frozen=True)
class TilePlan:
    """Per-(layer, head) tile shapes for one geometry.

    `head_shape[layer][head]` indexes into `shapes`, at most two distinct shapes per
    layer. The shape a head uses decides the permutation that head's tiles get, so
    the permutation is part of the plan and not a detail of the kernel.
    """

    geometry: str
    grid: tuple[int, int, int]
    shapes: tuple[TileShape, ...]
    head_shape: tuple[tuple[int, ...], ...]

    @classmethod
    def from_json(cls, data: dict) -> "TilePlan":
        return cls(
            geometry=data["geometry"],
            grid=tuple(data["grid"]),
            shapes=tuple(TileShape.parse(s) for s in data["shapes"]),
            head_shape=tuple(tuple(int(i) for i in row) for row in data["head_shape"]),
        )

    def uniform(self, geometry: str, grid: tuple[int, int, int]) -> "TilePlan":
        """One shape for every head of every layer; the no-search fallback."""
        return TilePlan(
            geometry=geometry,
            grid=grid,
            shapes=(least_padding_shape(grid),),
            head_shape=tuple(tuple(0 for _ in row) for row in self.head_shape),
        )

    def groups(self, layer: int) -> list[tuple[TileShape, np.ndarray]]:
        """(shape, head ids) of one layer, heads ascending, at most two entries."""
        row = np.asarray(self.head_shape[layer], dtype=np.int64)
        return [
            (self.shapes[int(index)], np.flatnonzero(row == index).astype(np.int32))
            for index in sorted({int(i) for i in row})
        ]


class PlanTable:
    """The plans a bundle carries, and the rule that answers any geometry."""

    def __init__(self, plans: Sequence[TilePlan]) -> None:
        self.plans = {plan.geometry: plan for plan in plans}

    def select(self, width: int, height: int, grid: tuple[int, int, int]) -> tuple[TilePlan, str]:
        """(plan, note) for a live video grid. Warn-and-go, never refuse.

        A plan is a suggestion table, not a contract: its tile shapes tile any grid,
        they just pad more outside the geometry they were searched on. So the answer
        to an untrained canvas is always "the nearest thing plus an honest note".
        """
        aspect = aspect_key(width, height)
        name = f"{aspect}_t{grid[0]}"
        plan = self.plans.get(name)
        if plan is not None and plan.grid == grid:
            return plan, f"plan {name} (exact)"
        same = [p for p in self.plans.values() if p.geometry.split("_t")[0] == aspect]
        if same:
            plan = min(same, key=lambda p: (abs(p.grid[0] - grid[0]), p.grid[0]))
            pad = plan_padding(plan, grid)
            return plan, (
                f"no plan for {name} grid {grid}; using {plan.geometry} (nearest latent_t of "
                f"aspect {aspect}), tiles pad {pad:.0%} over the live grid"
            )
        h, w = aspect.split("x")
        mirror = [p for p in self.plans.values() if p.geometry.split("_t")[0] == f"{w}x{h}"]
        if mirror:
            nearest = min(mirror, key=lambda p: abs(p.grid[0] - grid[0]))
            plan = TilePlan(name, grid, tuple(s.transposed() for s in nearest.shapes), nearest.head_shape)
            pad = plan_padding(plan, grid)
            return plan, f"no plan for aspect {aspect}; using the transposed plan of {nearest.geometry} (pads {pad:.0%})"
        template = next(iter(self.plans.values()))
        plan = template.uniform(name, grid)
        return plan, f"no plan for aspect {aspect}; uniform tile shape {plan.shapes[0]} over grid {grid}"


# --- the predictor bundle -------------------------------------------------


def _e4m3_table() -> np.ndarray:
    """Decode table for float8_e4m3fn: index is the raw byte, value is fp32.

    Re-derived rather than pulled from a library, because MLX exposes no fp8 dtype
    in this build and because the exporter quantized each head against 448.0, the
    largest finite magnitude of this exact format.
    """
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent, mantissa = (byte >> 3) & 0x0F, byte & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            table[byte] = math.nan
        elif exponent == 0:
            table[byte] = sign * (mantissa / 8.0) * 2.0**-6
        else:
            table[byte] = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return table


def _bf16_to_f32(raw: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """bf16 bytes -> fp32 by widening the mantissa, never through a lossy cast."""
    bits = raw.view(np.uint16).reshape(shape).astype(np.uint32)
    return (bits << 16).view(np.float32)


@dataclasses.dataclass(frozen=True)
class Bundle:
    """The predictor: two projections per (layer, head), plus its plan table."""

    proj_q: tuple[mx.array, ...]  # [heads, 3*dim, dim] fp32 per layer
    proj_k: tuple[mx.array, ...]
    plans: PlanTable
    keep_ratio: float
    dim: int
    metadata: dict

    @property
    def num_layers(self) -> int:
        return len(self.proj_q)

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        """Read a `miowtion-veda-predictor-v1` safetensors bundle.

        fp8 in the file, fp32 in memory: the projections feed fp32 matmuls and the
        per-head scale the exporter applied is applied back here. Half a GiB resident
        at the real shape, which on any machine that runs this trunk is noise, so
        the resident form is the accurate one.
        """
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no predictor bundle at {path}")
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
            meta = header.pop("__metadata__", None) or {}
            payload = np.frombuffer(f.read(), dtype=np.uint8)
        if meta.get("format") != "miowtion-veda-predictor-v1":
            raise ValueError(f"{path}: not a veda predictor bundle (format {meta.get('format')!r})")
        table = _e4m3_table()
        weights: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        for name, entry in header.items():
            begin, stop = (int(value) for value in entry["data_offsets"])
            raw = payload[begin:stop]
            shape = tuple(entry["shape"])
            if entry["dtype"] == "F32":
                values = raw.view(np.float32).reshape(shape)
            elif entry["dtype"] == "BF16":
                values = _bf16_to_f32(raw, shape)
            elif entry["dtype"] == "F8_E4M3":
                values = table[raw.reshape(shape)].astype(np.float32)
            else:
                raise ValueError(f"{path}: unsupported predictor dtype {entry['dtype']}")
            if name.endswith(_SCALE):
                scales[name[: -len(_SCALE)]] = values
            else:
                weights[name] = values.astype(np.float32)
        layers, heads, dim = int(meta["num_layers"]), int(meta["num_heads"]), int(meta["head_dim"])
        names = [f"layers.{i}.proj_{which}" for i in range(layers) for which in ("q", "k")]
        missing = sorted(set(names) - set(weights))
        if missing:
            raise ValueError(f"{path}: predictor is missing {len(missing)} tensors, e.g. {missing[:3]}")
        orphans = sorted(set(weights) - set(names))
        if orphans:
            raise ValueError(f"{path}: {len(orphans)} tensors match no predictor parameter, e.g. {orphans[:3]}")
        unsealed = sorted(set(n for n in names if weights[n].dtype == np.float32 and (n not in scales) and _fp8(header, n)))
        if unsealed:
            raise ValueError(f"{path}: fp8 tensors without a {_SCALE} scale, e.g. {unsealed[:3]}")
        for name in names:
            if tuple(weights[name].shape) != (heads, 3 * dim, dim):
                raise ValueError(f"{path}: {name} is {weights[name].shape}, want {(heads, 3 * dim, dim)}")
            if name in scales:
                weights[name] = weights[name] * scales[name].reshape(-1, 1, 1)
        plans = PlanTable(
            [TilePlan.from_json(data) for data in json.loads(meta["plans"]).values()] if meta.get("plans") else []
        )
        if not plans.plans:
            raise ValueError(f"{path}: bundle carries no tile plans")
        proj_q = tuple(mx.array(weights[f"layers.{i}.proj_q"]) for i in range(layers))
        proj_k = tuple(mx.array(weights[f"layers.{i}.proj_k"]) for i in range(layers))
        mx.eval(*proj_q, *proj_k)
        return cls(
            proj_q=proj_q,
            proj_k=proj_k,
            plans=plans,
            keep_ratio=float(meta.get("keep_ratio", 0.1)),
            dim=dim,
            metadata=meta,
        )


def _fp8(header: dict, name: str) -> bool:
    return header.get(name, {}).get("dtype") == "F8_E4M3"


# --- tile layout ----------------------------------------------------------


def span_tiles(grid: tuple[int, int, int], shape: TileShape, start: int) -> np.ndarray:
    """[n_tiles, TILE] packed row ids of one span, -1 on padding slots.

    Tile order is (h block, w block, t block) outer to inner; inside a tile the real
    rows are stably compacted to the front, so a tile's real rows are always a
    prefix of length `valid_count[tile]` and the kernel needs no per-row mask.
    """
    t, h, w = grid
    st, sh, sw = shape.t, shape.h, shape.w
    tp, hp, wp = shape.padded(grid)
    boxes = np.full((tp, hp, wp), -1, dtype=np.int64)
    boxes[:t, :h, :w] = start + np.arange(t * h * w).reshape(t, h, w)
    tiles = boxes.reshape(tp // st, st, hp // sh, sh, wp // sw, sw).transpose(2, 4, 0, 1, 3, 5)
    tiles = tiles.reshape(-1, TILE)
    order = np.argsort(tiles < 0, axis=1, kind="stable")
    return np.take_along_axis(tiles, order, axis=1)


@dataclasses.dataclass(frozen=True)
class HeadGroup:
    """One tile shape and the heads that use it, as ready-made MLX indices.

    Attributes:
        heads: global head ids, ascending; NOT assumed contiguous.
        gather: [N] packed row of every slot, padding slots redirected to row 0.
        inv_video: [video rows] the slot holding that packed video row.
        valid_count: [n_tiles] real rows per tile, fp32 so it can divide.
        n_video: leading tiles are the video quadrant; the rest are global.
    """

    shape: TileShape
    heads: tuple[int, ...]
    grid: tuple[int, int, int]
    video_start: int
    gather: mx.array
    head_ids: mx.array
    inv_video: mx.array
    valid_count: mx.array
    kv_ok: mx.bool_
    slot_valid: mx.bool_
    n_video: int
    n_tiles: int

    @property
    def video_tokens(self) -> int:
        return int(math.prod(self.grid))

    @property
    def n_global(self) -> int:
        return self.n_tiles - self.n_video

    @classmethod
    def build(cls, shape: TileShape, heads: Sequence[int], grid: tuple[int, int, int], video_start: int) -> "HeadGroup":
        heads = tuple(int(h) for h in sorted(heads))
        if not heads:
            raise ValueError("a head group needs at least one head")
        video = span_tiles(grid, shape, video_start)
        globals_ = np.arange(video_start, dtype=np.int64)
        n_global = int(math.ceil(globals_.size / TILE))
        global_tiles = np.full((n_global, TILE), -1, dtype=np.int64)
        global_tiles.reshape(-1)[: globals_.size] = globals_
        perm = np.concatenate([video.reshape(-1), global_tiles.reshape(-1)])
        valid = perm >= 0
        counts = valid.reshape(-1, TILE).sum(axis=1).astype(np.float32)
        inv = np.zeros(video_start + int(math.prod(grid)), dtype=np.int32)
        inv[perm[valid]] = np.flatnonzero(valid).astype(np.int32)
        return cls(
            shape=shape,
            heads=heads,
            grid=grid,
            video_start=video_start,
            gather=mx.array(np.where(valid, perm, 0), dtype=mx.int32),
            head_ids=mx.array(np.asarray(heads, dtype=np.int32)),
            inv_video=mx.array(inv[video_start:], dtype=mx.int32),
            valid_count=mx.array(counts),
            kv_ok=mx.array(counts > 0),
            slot_valid=mx.array(valid),
            n_video=video.shape[0],
            n_tiles=video.shape[0] + n_global,
        )


# --- selection -----------------------------------------------------------


def budget_per_row(keep_ratio: float, video_tokens: int, n_video: int) -> float:
    """Key tiles kept per query tile, at equal kernel cost.

    A padded grid has more tiles than real tokens / TILE. Paying for the padding with
    a bigger budget would quietly make a padded shot denser than an unpadded one, so
    the budget is scaled from the ideal tile count of the REAL tokens.
    """
    n_ideal = math.ceil(video_tokens / TILE)
    return keep_ratio * n_ideal * n_ideal / max(n_video, 1)


def dense_keep(video_tokens: int, n_video: int) -> float:
    """Smallest keep_ratio that leaves no video key column unvisited.

    Diagnostic knob, not a tuning knob: at this ratio every video query tile sees every
    video key tile, so the sparse path computes the dense softmax through the gathered
    layout. It costs MORE than dense and proves only that the gather/scatter/head
    plumbing round-trips. Compare it with a dense render at the same seed.
    """
    return (n_video / math.ceil(video_tokens / TILE)) ** 2


def columns_kept(keep_ratio: float, video_tokens: int, n_video: int) -> tuple[int, int]:
    """(video key columns a query tile sees, video key columns there to see).

    The honest coverage number. keep_ratio is a fraction of the IDEAL dense work, not
    of the columns: on a padded grid `n_video` exceeds `ceil(tokens / TILE)`, so the
    columns kept sit below keep_ratio and keep_ratio 1.0 is NOT dense. At the real
    864x480/56f grid the plan pads 56%, and keep 1.0 buys 42% of the columns.
    """
    k_lo, k_hi, _ = split_budget(budget_per_row(keep_ratio, video_tokens, n_video), n_video)
    return k_hi, n_video


def split_budget(budget: float, n_cols: int) -> tuple[int, int, float]:
    """(k_lo, k_hi, frac): rows keep k_lo or k_hi tiles, `frac` of them k_hi."""
    if budget >= n_cols:
        return n_cols, n_cols, 0.0
    k_lo = min(max(1, math.floor(budget)), n_cols)
    frac = round(min(max(budget - k_lo, 0.0), 1.0), 12)
    return k_lo, min(k_lo + 1, n_cols), frac


def bresenham(n_rows: int, frac: float) -> np.ndarray:
    """[n_rows] int32, 1 on the rows that get the extra tile. Host-side and exact."""
    ramp = np.floor(np.arange(n_rows + 1, dtype=np.float64) * frac)
    return (ramp[1:] > ramp[:-1]).astype(np.int32)


def select_tiles(logits: mx.array, group: HeadGroup, keep_ratio: float) -> tuple[mx.array, mx.array]:
    """(index, keep) of video key tiles per video query tile.

    `logits` is [heads, n_video, n_video]. The diagonal is forced in and consumes
    budget, empty tiles are never selected, the fractional budget is spread by
    Bresenham over query tiles, and the kept tiles are re-sorted by tile id so the
    gathered key order follows the packed order instead of the score order.
    """
    n_video = group.n_video
    # The predictor scores every tile against every tile; selection is the video
    # quadrant's business only. Global keys are dense by rule and global queries
    # never come here, so the square that matters is [:n_video, :n_video].
    logits = logits[:, :n_video, :n_video]
    k_lo, k_hi, frac = split_budget(budget_per_row(keep_ratio, group.video_tokens, n_video), n_video)
    diagonal = np.zeros((n_video, n_video), dtype=np.float32)
    np.fill_diagonal(diagonal, float("inf"))
    idx = mx.argsort(-(logits + mx.array(diagonal)), axis=-1)[..., :k_hi]
    allowed = mx.array(k_lo + bresenham(n_video, frac))
    keep = mx.arange(k_hi, dtype=mx.int32)[None, None, :] < allowed[None, :, None]
    keep = keep & mx.take(group.kv_ok[:n_video], idx, axis=0)
    order = mx.argsort(idx, axis=-1)
    return mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(keep, order, axis=-1)


# --- the attention itself ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LayerAttention:
    """One trunk layer's sparse attention: its groups, its predictor, its chunks."""

    groups: tuple[HeadGroup, ...]
    proj_q: mx.array  # [heads, 3*dim, dim] fp32
    proj_k: mx.array
    keep_ratio: float
    dim: int
    head_chunk: int = HEAD_CHUNK
    q_chunk: int = Q_CHUNK

    @staticmethod
    def _pool(tiled: mx.array, group: HeadGroup) -> mx.array:
        """[n, TILE, Hc, D] -> [Hc, n, 3D] fp32 [mean | max | min] over real rows.

        Padding slots hold zero, which is a legal value, so max and min are masked.
        An empty tile then yields -inf / +inf and the kv_ok select replaces it; a
        multiply there would give NaN.
        """
        valid = group.slot_valid.reshape(group.n_tiles, TILE, 1, 1)
        mean = tiled.astype(mx.float32).sum(axis=1) / group.valid_count[:, None, None]
        big = mx.array(float("inf"), dtype=tiled.dtype)
        tmax = mx.where(valid, tiled, -big).max(axis=1)
        tmin = mx.where(valid, tiled, big).min(axis=1)
        feats = mx.concatenate([mean, tmax.astype(mx.float32), tmin.astype(mx.float32)], axis=-1)
        return mx.where(group.kv_ok[:, None, None], feats, 0.0).transpose(1, 0, 2)

    def _chunk(self, q: mx.array, k: mx.array, v: mx.array, group: HeadGroup, heads: mx.array) -> mx.array:
        """Gathered sparse attention for one head chunk, output in packed row order."""
        scale = self.dim**-0.5
        q_g, k_g, v_g = (mx.take(t, heads, axis=1) for t in (q, k, v))
        alive = group.slot_valid[:, None, None]
        q_t, k_t, v_t = (
            mx.where(alive, mx.take(t, group.gather, axis=0), mx.array(0.0, dtype=t.dtype)).reshape(
                group.n_tiles, TILE, -1, self.dim
            )
            for t in (q_g, k_g, v_g)
        )
        feats_q, feats_k = (self._pool(t, group) for t in (q_t, k_t))
        # The predictor is indexed by GLOBAL head id, so the chunk's ids address it
        # directly -- no group-relative renumbering to get wrong.
        proj_q, proj_k = mx.take(self.proj_q, heads, axis=0), mx.take(self.proj_k, heads, axis=0)
        q_hat = feats_q @ proj_q + feats_q[..., : self.dim]
        k_hat = feats_k @ proj_k + feats_k[..., : self.dim]
        logits = (q_hat @ k_hat.transpose(0, 2, 1)) / math.sqrt(self.dim)
        idx, keep = select_tiles(logits, group, self.keep_ratio)

        # Global columns are dense by rule and never compete for budget.
        tiles_all = mx.arange(group.n_video, group.n_tiles, dtype=mx.int32)
        idx = mx.concatenate([idx, mx.broadcast_to(tiles_all[None, None, :], (idx.shape[0], idx.shape[1], group.n_global))], axis=-1)
        keep = mx.concatenate(
            [keep, mx.broadcast_to(group.kv_ok[group.n_video :][None, None, :], (idx.shape[0], idx.shape[1], group.n_global))],
            axis=-1,
        )
        counts = mx.take(group.valid_count, idx, axis=0)
        # Slot ids in the tile permutation, NOT packed rows: the permutation is
        # n_tiles*TILE long and the sequence is shorter than that. `gather` turns a
        # slot into the packed row it holds, padding slots into row 0 (and `valid`
        # is what keeps row 0 from being attended by mistake).
        slots = (idx[..., None] * TILE + mx.arange(TILE, dtype=mx.int32)[None, None, None, :]).reshape(idx.shape[0], idx.shape[1], -1)
        valid = (keep[..., None] & (mx.arange(TILE, dtype=mx.float32)[None, None, None, :] < counts[..., None])).reshape(
            idx.shape[0], idx.shape[1], -1
        )
        rows = mx.take(group.gather, mx.reshape(slots, (-1,)), axis=0).reshape(slots.shape)

        parts = []
        seq, h = q_g.shape[0], heads.shape[0]
        # Keys are gathered per head, so the index has to address a head-major flat
        # array: row + head*seq. Taking k_g directly with a 3-D index would broadcast
        # the head axis in and hand the kernel five dimensions.
        head_off = (mx.arange(h, dtype=mx.int32) * seq)[None, :, None]
        k_flat = mx.reshape(mx.transpose(k_g, (1, 0, 2)), (-1, self.dim))
        v_flat = mx.reshape(mx.transpose(v_g, (1, 0, 2)), (-1, self.dim))
        for start in range(0, group.n_video, self.q_chunk):
            stop = min(start + self.q_chunk, group.n_video)
            rows_c = rows[:, start:stop].transpose(1, 0, 2)  # [B, Hc, keys]
            flat = mx.reshape(rows_c + head_off, (-1,))
            mask = valid[:, start:stop].transpose(1, 0, 2)[:, :, None, :]  # [B, Hc, 1, keys]
            out = mx.fast.scaled_dot_product_attention(
                q_t[start:stop].transpose(0, 2, 1, 3),
                mx.take(k_flat, flat, axis=0).reshape(rows_c.shape + (self.dim,)),
                mx.take(v_flat, flat, axis=0).reshape(rows_c.shape + (self.dim,)),
                scale=scale,
                mask=mask,
            )
            mx.eval(out)
            parts.append(mx.transpose(out, (0, 2, 1, 3)))
        video = mx.concatenate(parts, axis=0).reshape(-1, q_g.shape[1], self.dim)
        return mx.take(video, group.inv_video, axis=0)

    def attention(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """q, k, v [seq, heads, dim] -> out [seq, heads, dim], sparse over video.

        Rows before the video span run dense against the whole sequence; the video
        span runs gathered. The two are joined by concatenation and one head gather,
        because MLX has no in-place scatter and a lazy graph must not hold every
        layer's intermediates alive at once.
        """
        scale = self.dim**-0.5
        pieces: list[mx.array] = []
        for group in self.groups:
            heads = list(group.heads)
            chunks = []
            for start in range(0, len(heads), self.head_chunk):
                ids = mx.array(np.asarray(heads[start : start + self.head_chunk], dtype=np.int32))
                video = self._chunk(q, k, v, group, ids)
                dense_q = mx.transpose(q[: group.video_start][:, ids, :], (1, 0, 2))[None]
                dense_k = mx.transpose(k[:, ids, :], (1, 0, 2))[None]
                dense_v = mx.transpose(v[:, ids, :], (1, 0, 2))[None]
                glob = mx.fast.scaled_dot_product_attention(dense_q, dense_k, dense_v, scale=scale)
                chunks.append(mx.concatenate([mx.transpose(glob, (0, 2, 1, 3))[0], video], axis=0))
                mx.eval(chunks[-1])
            pieces.append(mx.concatenate(chunks, axis=1))
            mx.eval(pieces[-1])
        out = mx.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
        order = np.argsort(np.concatenate([np.asarray(g.heads) for g in self.groups]))
        return mx.take(out, mx.array(order.astype(np.int32)), axis=1)


@dataclasses.dataclass(frozen=True)
class SparseTable:
    """The whole trunk's sparse configuration, built once per generation."""

    blocks: tuple[LayerAttention | None, ...]
    geometry: str
    note: str
    keep_ratio: float
    dense_layers: tuple[int, ...] = ()

    @property
    def sparse_layers(self) -> int:
        return sum(block is not None for block in self.blocks)

    def kept_tiles(self) -> tuple[int, int]:
        """(key tiles a query tile sees, key tiles there are) for the first sparse layer.

        Globals are counted on BOTH sides so the two numbers are comparable and the
        first is never larger than the second. The old version added the globals twice
        and reported ceil(budget), so it could claim more kept than there were.
        """
        for block in self.blocks:
            if block is not None:
                group = block.groups[0]
                kept, video = columns_kept(block.keep_ratio, group.video_tokens, group.n_video)
                return kept + group.n_global, video + group.n_global
        return 0, 0

    def coverage(self) -> tuple[float, float]:
        """(worst column coverage, keep_ratio needed to reach 100%).

        Reported because `keep 100%` in the banner would be a lie on any padded grid:
        the budget is a fraction of the ideal dense work, and padding spends part of it
        on tiles that hold no tokens. Coverage is the minimum over every sparse layer's
        head groups, since the tightest group is the one that loses context first.
        """
        worst, need = 1.0, 0.0
        for block in self.blocks:
            if block is None:
                continue
            for group in block.groups:
                kept, video = columns_kept(block.keep_ratio, group.video_tokens, group.n_video)
                worst = min(worst, kept / video)
                need = max(need, dense_keep(group.video_tokens, group.n_video))
        return worst, need



def build_table(
    bundle: Bundle,
    *,
    width: int,
    height: int,
    grid: tuple[int, int, int],
    video_start: int,
    num_layers: int,
    heads: int,
    keep_ratio: float | None = None,
    dense_layers: Sequence[int] = (),
    head_chunk: int = HEAD_CHUNK,
    q_chunk: int = Q_CHUNK,
) -> tuple[SparseTable, list[str]]:
    """Resolve a bundle against one packed sequence. Warn-and-go, never refuse.

    `grid` is the target video token grid (latent_t, latent_h // 2, latent_w // 2)
    and `video_start` the packed row where the target span begins; every row before
    it -- text, audio, keyframe anchors, references -- is global and stays dense.
    """
    if video_start < 1:
        raise ValueError("the video span must not start at row 0; text and audio precede it")
    warnings: list[str] = []
    plan, note = bundle.plans.select(width, height, grid)
    if not note.endswith("(exact)"):
        warnings.append(note)
    # Every operator knob normalizes here instead of refusing, and says what it landed on.
    # An out-of-range value costs a wasted arm; a refuse costs the whole batch, and the
    # operator cannot adjust what would not run.
    keep = bundle.keep_ratio
    if keep_ratio is not None:
        try:
            asked = float(keep_ratio)
        except (TypeError, ValueError):
            asked = None
        if asked is None or not math.isfinite(asked) or asked <= 0.0:
            warnings.append(f"keep_ratio {keep_ratio!r} is not a positive number -- the bundle's "
                            f"own {bundle.keep_ratio:g} is used instead, and the render goes ahead")
        else:
            keep = asked
    dense, dropped = [], []
    for layer in dense_layers:
        try:
            index = int(layer)
        except (TypeError, ValueError):
            dropped.append(repr(layer))
            continue
        if 0 <= index < num_layers:
            dense.append(index)
        else:
            dropped.append(index)
    dense = tuple(sorted(set(dense)))
    if dropped:
        warnings.append(f"dense_layers {[str(v) for v in dropped]} are not layer indices inside "
                        f"[0, {num_layers}) -- dropped, those layers run as the plan says")
    try:
        chunk = int(head_chunk)
    except (TypeError, ValueError):
        chunk = int(heads)
    if chunk < 1:
        # Only the low end matters: range() refuses 0, and a chunk wider than the head
        # count is harmless -- it just means one chunk. No upper gate, nya.
        warnings.append(f"head_chunk {head_chunk!r} is below 1 -- clamped to 1; it only "
                        f"bounds peak memory, never the result")
        chunk = 1
    blocks: list[LayerAttention | None] = []
    for layer in range(num_layers):
        if layer in dense:
            blocks.append(None)
            continue
        groups: list[HeadGroup] = []
        # A shape used by more heads than the chunk size is split. Selection is per
        # head anyway, so chunking a group costs nothing and bounds the gathered
        # key block, which is the only thing here that grows with the canvas.
        for shape, ids in plan.groups(layer % len(plan.head_shape)):
            for start in range(0, ids.size, chunk):
                part = ids[start : start + chunk]
                if part.size:
                    groups.append(HeadGroup.build(shape, part, grid, video_start))
        if not groups:
            raise ValueError(f"plan {plan.geometry} assigns no heads to layer {layer}")
        blocks.append(
            LayerAttention(
                groups=tuple(groups),
                proj_q=bundle.proj_q[layer],
                proj_k=bundle.proj_k[layer],
                keep_ratio=keep,
                dim=bundle.dim,
                head_chunk=chunk,
                q_chunk=q_chunk,
            )
        )
    table = SparseTable(blocks=tuple(blocks), geometry=plan.geometry, note=note, keep_ratio=keep, dense_layers=dense)
    # Say what the budget actually buys, never what keep_ratio literally spells: on a
    # padded grid the two differ, and an operator bisecting a bad render needs the
    # column number, not the ratio, to know whether a rung tested anything.
    cover, need = table.coverage()
    if cover >= 1.0:
        if keep >= 1.0:
            # The rung that bisects plumbing from prediction: every video query tile sees
            # every video key tile, so the pixels must match the dense control at the same
            # seed. Costs more than dense, and says so.
            warnings.append(
                f"keep_ratio {keep} keeps EVERY video key column: this run is a plumbing test, "
                f"mathematically dense attention through the gather path (slower than dense)"
            )
    elif keep >= 1.0 or keep - cover > 0.02:
        warnings.append(
            f"keep_ratio {keep} reaches only {cover:.0%} of video key columns, not {keep:.0%} "
            f"-- the budget is a fraction of the ideal dense work and this grid pads its "
            f"tiles ({plan_padding(plan, grid):.0%} over the live grid); every column needs "
            f"keep_ratio >= {need:.2f}"
        )
    return table, warnings
