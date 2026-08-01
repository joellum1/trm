# tests/test_transplant.py
import os
import torch
import pytest

from trm.diagnostics.transplant import (
    assert_charset_prefix,
    transplant_vocab,
    find_vocab_keys,
    VOCAB_KEY_PATTERNS,
)
from trm.data.custom import CHARSET

# Frozen literal, NOT imported. This is the vocabulary the pretrained
# maze checkpoint was trained under; it is a historical fact and must
# never change. Importing it would let the test drift with the code.
PRETRAINED_CHARSET = "# SGo"
V_OLD = len(PRETRAINED_CHARSET) + 1   # 6  (PAD + 5 symbols)
V_NEW = len(CHARSET) + 1              # 11 (PAD + 10 symbols)


def test_charset_is_prefix_extended():
    """New tiles must be APPENDED so token ids 0..5 keep their meaning."""
    assert_charset_prefix(PRETRAINED_CHARSET, CHARSET)


def test_vocab_sizes_are_as_expected():
    assert V_OLD == 6
    assert V_NEW == 11
    assert CHARSET[len(PRETRAINED_CHARSET):] == "CRrNn"


def test_old_rows_preserved_exactly():
    sd = {
        "model.inner.embed_tokens.embedding_weight": torch.randn(V_OLD, 512),
        "model.inner.lm_head.weight": torch.randn(V_OLD, 512),
        "model.inner.q_head.weight": torch.randn(2, 512),
    }
    keys = [
        "model.inner.embed_tokens.embedding_weight",
        "model.inner.lm_head.weight",
    ]
    out = transplant_vocab(sd, keys, V_OLD, V_NEW)
    for k in keys:
        assert out[k].shape[0] == V_NEW
        assert torch.equal(out[k][:V_OLD], sd[k]), f"old rows corrupted in {k}"
    assert torch.equal(
        out["model.inner.q_head.weight"], sd["model.inner.q_head.weight"]
    ), "q_head must not be touched"


def test_new_rows_are_small():
    sd = {"model.inner.embed_tokens.embedding_weight": torch.randn(V_OLD, 512) * 3.0}
    k = "model.inner.embed_tokens.embedding_weight"
    out = transplant_vocab(sd, [k], V_OLD, V_NEW)
    new_max = out[k][V_OLD:].abs().max()
    old_max = sd[k].abs().max()
    assert 0 < new_max < old_max * 0.2, "new rows must be nonzero but small"


def test_find_vocab_keys_matches_by_name_and_shape():
    sd = {
        "model.inner.embed_tokens.embedding_weight": torch.randn(V_OLD, 512),
        "model.inner.lm_head.weight": torch.randn(V_OLD, 512),
        "model.inner.q_head.weight": torch.randn(2, 512),
        "model.inner.puzzle_emb.weights": torch.randn(1, 512),
        "model.inner.L_level.layers.0.mlp.down_proj.weight": torch.randn(512, 1536),
    }
    keys = set(find_vocab_keys(sd, V_OLD))
    assert keys == {
        "model.inner.embed_tokens.embedding_weight",
        "model.inner.lm_head.weight",
    }, f"unexpected key selection: {keys}"


CKPT = os.environ.get("TRM_PRETRAINED_CKPT", "")


@pytest.mark.skipif(not CKPT or not os.path.exists(CKPT),
                    reason="set TRM_PRETRAINED_CKPT to the maze_hard checkpoint")
def test_real_checkpoint_inventory():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    keys = find_vocab_keys(sd, V_OLD)
    print("\nvocab-dependent tensors:")
    for k in keys:
        print(f"  {k}  {tuple(sd[k].shape)}")
    assert len(keys) == 2, f"expected embed + lm_head, got {keys}"
    for k in keys:
        assert sd[k].shape[0] == V_OLD