"""The Veda tile path, against the definition of itself.

WHAT THESE PROVE. Sparse attention fails in exactly one way that matters: it drops a
key tile some head needed, and the picture quietly loses the thing that key was
holding. So these tests do not ask "did it run" -- they ask "is the gathered answer
the same answer a masked dense call would have given, tile for tile, head for head",
plus the index algebra that makes that claim meaningful: a tile's real rows are a
prefix, every real row lives in exactly one slot, and the budget means what it says.

WHAT THEY DO NOT PROVE. That the predictor's choices are good ones. That is a render
and an eye, and mmcat's `archive/sparse_attn_AB_2026-10-02.md` is the standing
reminder that a path can be exact, fast, and still lose the camera.

Small shapes on purpose: the gather path's cost is set by the budget, so a 4-head
fixture exercises the same code the 56-head trunk does. Only the constants change.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_h3 import sparse

HEADS, DIM = 4, 32
#: A target grid that tiles EXACTLY with (2x8x8): two tiles, no padding, so
#: keep_ratio=1.0 is a true "everything kept" case against dense.
GRID = (4, 8, 8)
SHAPE = sparse.TileShape(2, 8, 8)
VIDEO_START = 40  # text + audio + anchors: global rows, dense by rule
SEQ = VIDEO_START + math.prod(GRID)
SCALE = DIM**-0.5


@pytest.fixture(scope="module", autouse=True)
def on_cpu():
    """Pin the CPU stream. This Metal device rounds fp32 matmul inputs (~7.5e-4),
    two orders of magnitude above what a structural check needs."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _group(shape=SHAPE, heads=range(HEADS), grid=GRID, video_start=VIDEO_START):
    return sparse.HeadGroup.build(shape, list(heads), grid, video_start)


def _layer(keep_ratio=0.5, seed=0, head_chunk=HEADS, q_chunk=2, group=None):
    group = group or _group()
    rng = np.random.default_rng(seed)
    return sparse.LayerAttention(
        groups=(group,),
        proj_q=mx.array(rng.standard_normal((HEADS, 3 * DIM, DIM), dtype=np.float32) * 1e-4),
        proj_k=mx.array(rng.standard_normal((HEADS, 3 * DIM, DIM), dtype=np.float32) * 1e-4),
        keep_ratio=keep_ratio,
        dim=DIM,
        head_chunk=head_chunk,
        q_chunk=q_chunk,
    )


def _qa(seed=1):
    rng = np.random.default_rng(seed)
    return [mx.array(rng.standard_normal((SEQ, HEADS, DIM), dtype=np.float32)) for _ in range(3)]


def _tiled(t: mx.array, group) -> mx.array:
    """The group's head rows in tile order, padding slots zeroed."""
    return mx.where(
        group.slot_valid[:, None, None],
        mx.take(mx.take(t, group.head_ids, axis=1), group.gather, axis=0),
        mx.array(0.0, dtype=t.dtype),
    ).reshape(group.n_tiles, sparse.TILE, -1, DIM)


def _logits(layer: sparse.LayerAttention, q, k) -> mx.array:
    group = layer.groups[0]
    feats_q, feats_k = (layer._pool(_tiled(t, group), group) for t in (q, k))
    proj_q = mx.take(layer.proj_q, group.head_ids, axis=0)
    proj_k = mx.take(layer.proj_k, group.head_ids, axis=0)
    q_hat = feats_q @ proj_q + feats_q[..., :DIM]
    k_hat = feats_k @ proj_k + feats_k[..., :DIM]
    return (q_hat @ k_hat.transpose(0, 2, 1)) * SCALE


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _plan(name: str, grid=(4, 8, 8), shapes=("2x8x8",), layers=2) -> sparse.TilePlan:
    return sparse.TilePlan.from_json(
        {
            "geometry": name,
            "grid": list(grid),
            "shapes": list(shapes),
            "head_shape": [[index % len(shapes) for index in range(HEADS)] for _ in range(layers)],
        }
    )


class _Bundle:
    """Just enough of a Bundle for build_table: the plan table and zero weights."""

    def __init__(self, plans: sparse.PlanTable, layers=4, heads=HEADS, keep=0.1):
        self.proj_q = tuple(mx.zeros((heads, 3 * DIM, DIM)) for _ in range(layers))
        self.proj_k = tuple(mx.zeros((heads, 3 * DIM, DIM)) for _ in range(layers))
        self.keep_ratio = keep
        self.dim = DIM
        self.num_layers = layers
        self.metadata = {"num_heads": heads}
        self.plans = plans


def _build(plans, layers=4, **kwargs):
    return sparse.build_table(
        _Bundle(plans, layers=layers),
        width=kwargs.pop("width", 16 * 8),
        height=kwargs.pop("height", 9 * 8),
        grid=kwargs.pop("grid", (4, 8, 8)),
        video_start=VIDEO_START,
        num_layers=layers,
        heads=HEADS,
        **kwargs,
    )


# --- format and plan fallbacks -------------------------------------------


def test_e4m3_table_is_the_format_the_bundle_was_quantized_against():
    table = sparse._e4m3_table()
    assert table[0] == 0.0 and table[0x80] == -0.0
    assert table[0x7E] == pytest.approx(448.0), "largest finite magnitude"
    assert np.isnan(table[0x7F]) and np.isnan(table[0xFF]), "the only NaN bit patterns"
    assert table[0x08] == pytest.approx(2.0**-6), "smallest normal"
    assert table[0x01] == pytest.approx(2.0**-9), "smallest subnormal"
    assert table[0x07] == pytest.approx(0.875 * 2.0**-6), "largest subnormal"


def test_plan_select_answers_every_geometry_without_refusing():
    table = sparse.PlanTable([_plan("16x9_t4"), _plan("4x3_t8", grid=(8, 8, 8), shapes=("1x8x16",))])

    plan, note = table.select(16 * 8, 9 * 8, (4, 8, 8))
    assert plan.geometry == "16x9_t4" and note.endswith("(exact)")

    plan, note = table.select(16 * 8, 9 * 8, (7, 8, 8))
    assert plan.geometry == "16x9_t4" and "nearest latent_t" in note, "same aspect, other length"

    plan, note = table.select(9 * 8, 16 * 8, (7, 8, 4))
    assert "transposed" in note and plan.shapes[0] == SHAPE.transposed(), "portrait borrows the mirror"

    plan, note = table.select(5 * 8, 7 * 8, (7, 10, 8))
    assert "transposed" in note and "3x4" in note, "an unsearched aspect borrows its mirror"
    assert plan.grid == (7, 10, 8) and plan.shapes[0] == sparse.TileShape(1, 16, 8)


def test_a_square_canvas_with_no_square_and_no_mirror_plan_falls_back_to_uniform():
    table = sparse.PlanTable([_plan("16x9_t4")])
    plan, note = table.select(8 * 8, 8 * 8, (7, 8, 8))
    assert "uniform" in note and plan.grid == (7, 8, 8)
    assert {tuple(row) for row in plan.head_shape} == {(0,) * HEADS}
    assert plan.shapes[0].t * plan.shapes[0].h * plan.shapes[0].w == sparse.TILE
    assert plan.shapes[0].num_tiles(plan.grid) * sparse.TILE >= math.prod(plan.grid)


def test_two_shapes_in_one_layer_become_two_groups_with_two_permutations():
    plan = _plan("16x9_t4", shapes=("2x8x8", "4x8x4"))
    plan = sparse.TilePlan.from_json(
        {"geometry": "16x9_t4", "grid": [4, 8, 8], "shapes": ["2x8x8", "4x8x4"], "head_shape": [[0, 1, 0, 1]]}
    )
    groups = plan.groups(0)
    assert [shape for shape, _ in groups] == [sparse.TileShape(2, 8, 8), sparse.TileShape(4, 8, 4)]
    assert [list(heads) for _, heads in groups] == [[0, 2], [1, 3]]

    table, notes = _build(sparse.PlanTable([plan]), layers=1)
    block = table.blocks[0]
    assert {group.heads for group in block.groups} == {(0, 2), (1, 3)}
    assert all(group.n_video > 0 and group.n_global > 0 for group in block.groups)
    assert notes == [], "16x9_t4 with the searched grid is exact, so nothing to warn about"


# --- tile layout algebra -----------------------------------------------


def test_span_tiles_keeps_real_rows_as_a_prefix_of_every_tile():
    tiles = sparse.span_tiles(GRID, SHAPE, VIDEO_START)
    assert tiles.shape == (2, sparse.TILE)
    for row in tiles:
        count = int((row >= 0).sum())
        assert np.all(row[:count] >= 0) and np.all(row[count:] < 0), "real rows lead"
    assert np.array_equal(np.sort(tiles[tiles >= 0]), np.arange(VIDEO_START, SEQ)), "every row exactly once"


def test_head_group_partitions_the_sequence_and_inverts_the_video_span():
    group = _group()
    assert group.n_tiles == 2 + math.ceil(VIDEO_START / sparse.TILE)
    assert float(group.valid_count.sum()) == SEQ, "each real row lives in exactly one slot"
    gathered = np.asarray(group.gather)
    assert np.array_equal(gathered[np.asarray(group.inv_video)], np.arange(VIDEO_START, SEQ)), "inv_video inverts gather"
    assert np.all(np.asarray(group.valid_count) > 0), "an empty tile would divide the pooled mean by zero"


def test_padding_slots_land_on_row_zero_and_are_marked_dead():
    group = _group(grid=(3, 7, 8))  # pads in every axis
    gathered, alive = np.asarray(group.gather), np.asarray(group.slot_valid)
    assert gathered[~alive].tolist() == [0] * int((~alive).sum()), "padding reads a real row, never -1"
    assert int(alive.sum()) == math.prod((3, 7, 8)) + VIDEO_START, "real video rows plus the global span"


# --- budget rules ----------------------------------------------------


def test_budget_scales_from_real_tokens_not_padded_tiles():
    grid, shape = (4, 8, 8), sparse.TileShape(1, 1, 128)
    tokens = math.prod(grid)
    assert sparse.budget_per_row(0.1, tokens, shape.num_tiles(grid)) == pytest.approx(
        0.1 * math.ceil(tokens / sparse.TILE) ** 2 / shape.num_tiles(grid)
    )
    assert sparse.split_budget(0.0, 10) == (1, 2, 0.0), "a zero budget still attends something"
    assert sparse.split_budget(99, 10) == (10, 10, 0.0), "an over-budget keeps everything"
    k_lo, k_hi, frac = sparse.split_budget(2.25, 40)
    assert (k_lo, k_hi, frac) == (2, 3, 0.25)
    assert int(sparse.bresenham(400, frac).sum()) == 100, "the fraction is exact in the mean"


def test_selection_forces_the_diagonal_and_matches_the_budget_exactly():
    group = _group()
    logits = mx.array(np.random.default_rng(7).standard_normal((HEADS, group.n_video, group.n_video), dtype=np.float32))
    for keep in (0.25, 0.5, 0.9, 1.0):
        idx, keep_mask = sparse.select_tiles(logits, group, keep_ratio=keep)
        idx_np, keep_np = np.asarray(idx), np.asarray(keep_mask)
        assert idx_np.shape == keep_np.shape
        for row in range(group.n_video):
            assert row in idx_np[:, row, :][keep_np[:, row, :]], f"row {row} lost its own tile at keep={keep}"
        allowed = _allowed_counts(group, keep)
        assert keep_np.sum(axis=-1).tolist() == [[int(count) for count in allowed[0]] for _ in range(HEADS)]


def _allowed_counts(group, keep_ratio) -> np.ndarray:
    k_lo, k_hi, frac = sparse.split_budget(sparse.budget_per_row(keep_ratio, group.video_tokens, group.n_video), group.n_video)
    return (k_lo + sparse.bresenham(group.n_video, frac))[None, :]


# --- the gathered answer is the masked answer ------------------------


def _reference(q, k, v, layer: sparse.LayerAttention) -> np.ndarray:
    """Dense attention under the block mask the plan implies, per head.

    Same keys, packed order, disallowed keys simply absent. The gather path
    reorders keys, so what has to agree is the softmax result, not any
    intermediate.
    """
    group = layer.groups[0]
    gathered, alive = np.asarray(group.gather), np.asarray(group.slot_valid)
    idx_np, keep_np = (np.asarray(a) for a in sparse.select_tiles(_logits(layer, q, k), group, layer.keep_ratio))
    q_np, k_np, v_np = (np.asarray(t) for t in (q, k, v))
    out = np.zeros((SEQ, HEADS, DIM), dtype=np.float32)

    # Global query rows: plain dense attention over the whole sequence.
    out[:VIDEO_START] = np.einsum("hqk,khd->qhd", _softmax(np.einsum("qhd,khd->hqk", q_np[:VIDEO_START], k_np) * SCALE), v_np)

    def rows_of(tile: int) -> np.ndarray:
        span = slice(tile * sparse.TILE, (tile + 1) * sparse.TILE)
        return gathered[span][alive[span]]

    # Global columns are dense for every video query row, exactly as the kernel appends them.
    globals_live = np.concatenate([rows_of(tile) for tile in range(group.n_video, group.n_tiles)])

    for tile in range(group.n_video):
        live = rows_of(tile)
        if live.size == 0:
            continue
        for head in range(HEADS):
            keys = np.concatenate(
                [rows_of(int(other)) for other, kept in zip(idx_np[head, tile], keep_np[head, tile]) if kept] + [globals_live]
            )
            assert keys.size > 0, "global columns are dense by rule, so no row starves"
            out[live, head] = _softmax(q_np[live, head] @ k_np[keys, head].T * SCALE) @ v_np[keys, head]
    return out


def test_gathered_attention_equals_masked_dense():
    q, k, v = _qa()
    layer = _layer(keep_ratio=0.5)
    got, want = np.asarray(layer.attention(q, k, v)), _reference(q, k, v, _layer(keep_ratio=0.5))
    unit = float(np.abs(want).mean())
    assert np.isfinite(got).all()
    video = float(np.abs(got[VIDEO_START:] - want[VIDEO_START:]).max()) / unit
    glob = float(np.abs(got[:VIDEO_START] - want[:VIDEO_START]).max()) / unit
    assert video < 1e-5, f"video rows drift {video:.2e} off the masked reference"
    assert glob < 1e-5, "global rows must be plain dense attention over everything"


def test_full_budget_is_dense_within_bf16_noise():
    q, k, v = _qa(seed=3)
    dense = mx.fast.scaled_dot_product_attention(
        mx.transpose(q, (1, 0, 2))[None],
        mx.transpose(k, (1, 0, 2))[None],
        mx.transpose(v, (1, 0, 2))[None],
        scale=SCALE,
    )[0]
    dense = mx.transpose(dense, (1, 0, 2))
    got = _layer(keep_ratio=1.0, seed=3).attention(q, k, v)
    ratio = float(mx.abs(got - dense).max() / mx.abs(dense).mean())
    assert ratio < 5e-3, f"full budget drifted {ratio:.2e} off dense"


def test_a_dropped_key_tile_is_invisible_to_the_rows_that_dropped_it():
    """The diagonal is forced, so "dropped" is always relative to a query tile:
    change a tile these rows never kept and their answer must not move; change it
    for the rows that DID keep it (its own rows) and they must move."""
    grid = (8, 8, 8)  # four video tiles, a quarter budget = exactly one tile a row
    group = _group(grid=grid)
    seq = VIDEO_START + math.prod(grid)
    rng = np.random.default_rng(5)
    q, k, v = (mx.array(rng.standard_normal((seq, HEADS, DIM), dtype=np.float32)) for _ in range(3))
    layer = _layer(keep_ratio=0.25, seed=5, group=group)
    keep_np = np.asarray(sparse.select_tiles(_logits(layer, q, k), group, layer.keep_ratio)[1])
    assert keep_np.sum() == HEADS * group.n_video, "a quarter budget keeps the diagonal and nothing else"

    other = slice(sparse.TILE, 2 * sparse.TILE)  # key tile 1, kept by tile 0's queries by nobody
    span = slice(0, sparse.TILE)  # tile 0's query rows
    live_other = np.asarray(group.gather)[other][np.asarray(group.slot_valid)[other]]
    rows_self = np.asarray(group.gather)[span][np.asarray(group.slot_valid)[span]]
    before = np.asarray(layer.attention(q, k, v))
    v2 = np.array(v)
    v2[live_other] += 10.0
    after = np.asarray(layer.attention(q, k, mx.array(v2)))
    drift = float(np.abs(after - before)[rows_self].max() / np.abs(before[rows_self]).mean())
    assert drift < 1e-6, f"a tile nobody kept for these rows moved them by {drift:.2e}"
    kept_moved = float(np.abs(after - before)[live_other].max())
    assert kept_moved > 0, "the rows that DO keep the tile must feel it, or nothing attends anything"


def test_head_and_query_chunking_is_a_memory_knob_not_a_computation():
    q, k, v = _qa(seed=11)
    wide = _layer(keep_ratio=0.5, seed=11)
    chunky = sparse.LayerAttention(
        groups=tuple(_group(heads=[head]) for head in range(HEADS)),
        proj_q=wide.proj_q,
        proj_k=wide.proj_k,
        keep_ratio=0.5,
        dim=DIM,
        head_chunk=1,
        q_chunk=1,
    )
    a, b = np.asarray(wide.attention(q, k, v)), np.asarray(chunky.attention(q, k, v))
    assert float(np.abs(a - b).max() / np.abs(a).mean()) < 1e-5


def test_dense_layers_stay_dense_and_are_named_in_the_table():
    table, notes = _build(sparse.PlanTable([_plan("16x9_t4")]), layers=4, dense_layers=(1, 2))
    assert [block is None for block in table.blocks] == [False, True, True, False]
    assert table.dense_layers == (1, 2) and table.sparse_layers == 2
    assert table.keep_ratio == 0.1 and notes == []
    kept, video_tiles = table.kept_tiles()
    assert kept > 0 and video_tiles == 2, "the receipt names the budget it will actually pay"


def test_keep_ratio_one_warns_that_sparse_is_a_no_op():
    table, notes = _build(sparse.PlanTable([_plan("16x9_t4")]), layers=1, keep_ratio=1.0)
    assert any("no-op" in note for note in notes), "a lever that does nothing says so"
    assert table.blocks[0].keep_ratio == 1.0


def test_an_unsearched_canvas_runs_and_says_what_it_borrowed():
    table, notes = _build(sparse.PlanTable([_plan("16x9_t4")]), layers=1, width=864, height=480, grid=(7, 15, 27))
    assert table.blocks[0] is not None and notes, "warn, and go"
    assert "nearest latent_t" in notes[0]
