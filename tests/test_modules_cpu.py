"""Shape and dtype checks for the memory modules on CPU with a small hidden size."""
import torch

from remap.modeling.fact_writer import canonicalize, question_stem, _slot_ranges
from remap.modeling.global_memory import GlobalMemory, partition
from remap.modeling.local_memory import LocalMemory, crop_region, pack_k32
from remap.modeling.model import MemoryTypeEmbedding
from remap.modeling.relation import RelationConfig, RelationProducer, grid_ids

D = 64


def test_partition():
    assert [partition(n)[-1].end for n in range(1, 11)] == [16, 16, 16, 20, 24, 28, 32, 36, 40, 44]


def test_global_memory():
    torch.manual_seed(0)
    memory, types = GlobalMemory(D, rank=8), MemoryTypeEmbedding(D)
    for count in (1, 2, 3, 4, 7):
        lengths = [5 + i for i in range(count)]
        visual = torch.randn(sum(lengths), D, dtype=torch.bfloat16)
        ids = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
        if count <= 3:
            raw = memory.raw_low_n(visual, ids, torch.randn(4, D, dtype=torch.bfloat16), count)
        else:
            summaries = torch.stack([visual[ids == i].float().mean(0) for i in range(count)]).bfloat16()
            raw = memory.raw_high_n(visual, ids, summaries, torch.randn(D), count)
        block = memory.bridge(raw, count, types)
        assert block.shape == (1, partition(count)[-1].end, D)
        if count > 3:
            assert memory.readout(block, torch.randn(1, D)).shape == block.shape


def test_local_memory():
    torch.manual_seed(0)
    local = LocalMemory(D, selector_rank=8, context_rank=8, conditioner_rank=8, set_context_rank=8)
    states = [torch.randn(1, 6 + i, D) for i in range(5)]
    selection = local.selector(torch.randn(1, D), states)
    assert 1 <= len(selection.positions) <= 3
    assert local.selector(torch.randn(1, D), states[:2]).positions == (0, 1)
    query = local.conditioner.condition(torch.randn(1, D), torch.randn(1, 16, D))
    blocks, readers = [], []
    for index in selection.positions:
        context = local.weaver.image_context(states, anchor_index=index, roi_image_index=index)
        block, reader = local.weaver(torch.randn(1, 9, D), query_state=query, bbox=torch.rand(1, 4),
                                     image_context=context)
        blocks.append(block)
        readers.append(reader)
    packed = pack_k32(blocks)
    if len(blocks) > 1:
        packed = local.conditioner.mix(blocks, [states[i] for i in selection.positions], query, packed)
    fused = local.set_context(packed, states)
    assert fused.shape == (1, 32, D)
    adapted = local.weaver.read_decoder_hidden(torch.randn(1, 1, D), pack_k32(readers))
    assert adapted.shape == (1, 1, D)


def test_relation():
    producer = RelationProducer(RelationConfig(hidden_size=D, rank=16, num_heads=4))
    grid = torch.tensor([[1, 4, 6], [1, 2, 2]])
    image_ids, rows, cols = grid_ids(grid, "cpu")
    out = producer(torch.randn(1, image_ids.numel(), D), image_ids=image_ids[None], row_ids=rows[None],
                   col_ids=cols[None], question_tokens=torch.randn(1, 7, D), num_images=2)
    assert out.shape == (1, 16, D)


def test_crop_and_text():
    from PIL import Image
    crop = crop_region(Image.new("RGB", (400, 300)), [10, 10, 20, 20])
    assert min(crop.size) >= 28
    assert question_stem("What color?\nA. red\nB. blue") == "What color?"
    fact = canonicalize("I1{a red cup}; I2{ a blue cup }; REL{same shape}", 2)
    assert fact == "I1{a red cup}; I2{a blue cup}; REL{same shape}."
    assert set(_slot_ranges(fact, 2)) == {"I1", "I2", "REL"}
