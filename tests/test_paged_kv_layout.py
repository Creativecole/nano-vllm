import torch

from nanovllm.kernels.attention import build_block_tables, pack_dense_kv_to_cache


def test_dense_kv_pack_uses_block_table_layout():
    batch_size, seq_len, block_size = 2, 5, 2
    num_kv_heads, head_dim = 2, 4
    dense_k = torch.arange(
        batch_size * seq_len * num_kv_heads * head_dim,
        dtype=torch.float32,
    ).reshape(batch_size, seq_len, num_kv_heads, head_dim)
    dense_v = dense_k + 1000
    block_tables = build_block_tables(batch_size, seq_len, block_size)

    k_cache, v_cache = pack_dense_kv_to_cache(dense_k, dense_v, block_tables, block_size)

    for batch_id in range(batch_size):
        for pos in range(seq_len):
            logical_block = pos // block_size
            block_offset = pos % block_size
            physical_block = int(block_tables[batch_id, logical_block])
            torch.testing.assert_close(k_cache[physical_block, block_offset], dense_k[batch_id, pos])
            torch.testing.assert_close(v_cache[physical_block, block_offset], dense_v[batch_id, pos])


def test_build_block_tables_assigns_disjoint_physical_blocks():
    tables = build_block_tables(batch_size=3, seq_len=9, block_size=4)
    assert tables.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]

