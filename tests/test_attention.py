import torch
from ltx_video.models.transformers.attention import Attention
import os
import subprocess
import tempfile
import pathlib

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _make_attention(query_dim: int = 32, heads: int = 4, dim_head: int = 8) -> Attention:
    """Return a deterministic `Attention` module in *eval* mode.

    Parameters
    ----------
    query_dim : int
        Embedding dimension of the query/key/value inputs.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head (inner dim = ``heads x dim_head``).
    """
    attn = Attention(
        query_dim=query_dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.0,  
        bias=True,
        use_rope=False,
    )
    attn.eval()
    return attn


def _expected_mask(q_seg: torch.Tensor, kv_seg: torch.Tensor) -> torch.Tensor:
    q_pad = q_seg == 0
    kv_pad = kv_seg == 0
    return ~(
        (q_seg.unsqueeze(-1) != kv_seg.unsqueeze(-2)) | (q_pad.unsqueeze(-1) | kv_pad.unsqueeze(-2))
    )

# -----------------------------------------------------------------------------
#  Tests
# -----------------------------------------------------------------------------

def test_prepare_attention_mask_diff_kv_q():
    """
    We compare the computed mask to a manually chosen mask.
    """
    kv_seg = torch.tensor([[1, 1, 1, 0, 2, 2]])  # (B=1, L_kv=6)
    q_seg = torch.tensor([[1, 0, 2]])            # (B=1, L_q =3)

    attn = _make_attention(query_dim=8, heads=1, dim_head=8)
    mask = attn.prepare_attention_mask(
        kv_segment_ids=kv_seg,
        q_segment_ids=q_seg,
        target_length=kv_seg.size(1),
        batch_size=1,
        out_dim=4,
    )

    expected = torch.tensor(
        [[[[ True,  True,  True, False, False, False],
           [False, False, False, False, False, False],
           [False, False, False, False,  True,  True]]]],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_prepare_attention_mask_batch():
    """
    This test checks that the mask computation broadcasts correctly over the batch axis.
    """
    kv_seg = torch.tensor(
        [
            [1, 1, 1, 0, 2, 2],   # B0
            [1, 0, 3, 3, 0, 3],   # B1
        ]
    )
    q_seg = kv_seg.clone()

    attn = _make_attention(query_dim=8, heads=1, dim_head=8)
    mask = attn.prepare_attention_mask(
        kv_segment_ids=kv_seg,
        q_segment_ids=q_seg,
        target_length=kv_seg.size(1),
        batch_size=kv_seg.size(0),
        out_dim=4,
    )

    expected = _expected_mask(q_seg, kv_seg).unsqueeze(1)  # (B, H, Lq, Lk)
    assert torch.equal(mask, expected)


def test_attention_segment_equivalence():
    """
    This test checks that the output of running attention on a padded sequence that contains two logical
    segments is the same as running attention separately for each segment.
    """
    torch.manual_seed(0)
    attn = _make_attention(query_dim=32, heads=4, dim_head=8)

    seg_comb = torch.tensor([[1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]])
    hidden_comb = torch.randn(1, seg_comb.size(1), 32)

    out_comb = attn(
        hidden_comb,
        hidden_states_segment_ids=seg_comb,
        encoder_hidden_states_segment_ids=seg_comb,
    )

    # Segment-1 check
    idx1 = seg_comb[0] == 1
    out1 = attn(
        hidden_comb[:, idx1, :],
        hidden_states_segment_ids=seg_comb[:, idx1],
        encoder_hidden_states_segment_ids=seg_comb[:, idx1],
    )
    assert torch.allclose(out1, out_comb[:, idx1, :], atol=1e-5, rtol=1e-4)

    # Segment-2 check
    idx2 = seg_comb[0] == 2
    out2 = attn(
        hidden_comb[:, idx2, :],
        hidden_states_segment_ids=seg_comb[:, idx2],
        encoder_hidden_states_segment_ids=seg_comb[:, idx2],
    )
    assert torch.allclose(out2, out_comb[:, idx2, :], atol=1e-5, rtol=1e-4)


def test_attention_none_vs_allones():
    """
    This test checks that the output of running attention with no segment IDs is the same as running attention with all ones as segment IDs.
    """
    torch.manual_seed(1)
    attn = _make_attention()
    hidden = torch.randn(1, 5, 32)

    out_none = attn(hidden)  # implicit unmasked path

    seg_ids = torch.ones(1, 5, dtype=torch.long)
    out_seg = attn(
        hidden,
        hidden_states_segment_ids=seg_ids,
        encoder_hidden_states_segment_ids=seg_ids,
    )

    assert torch.allclose(out_none, out_seg, atol=1e-5, rtol=1e-4)

# TODO: test tpu flash attention