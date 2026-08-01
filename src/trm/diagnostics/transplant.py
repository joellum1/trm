"""Vocabulary transplant for TRM checkpoints.

Extends the token embedding and output head from a smaller vocabulary to a
larger one, preserving the original rows byte-for-byte so a pretrained
checkpoint keeps its learned representation of the original symbols.

Safe ONLY when the new charset is a strict extension of the old one, i.e.
NEW_CHARSET[:len(OLD_CHARSET)] == OLD_CHARSET, so existing token ids do not
move. Call assert_charset_prefix() to enforce that.
"""

from typing import Dict, List, Sequence
import torch

# Tensors whose leading axis is the vocabulary, by name. Matching on name
# AND shape avoids accidentally grabbing an unrelated tensor that happens
# to have a dimension equal to the old vocab size.
VOCAB_KEY_PATTERNS: Sequence[str] = (
    "embed_tokens.embedding_weight",
    "lm_head.weight",
)


def assert_charset_prefix(old: str, new: str) -> None:
    assert new[:len(old)] == old, (
        "new tokens must be APPENDED, not inserted -- old token ids must not move.\n"
        f"  old            : {old!r}\n"
        f"  new[:{len(old)}]       : {new[:len(old)]!r}\n"
        "Fix CHARSET ordering and rebuild the dataset."
    )
    assert len(set(new)) == len(new), f"duplicate symbol in CHARSET: {new!r}"


def find_vocab_keys(sd: Dict[str, torch.Tensor], v_old: int) -> List[str]:
    """Return state-dict keys whose leading axis is the vocabulary."""
    out = []
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if any(k.endswith(p) for p in VOCAB_KEY_PATTERNS):
            assert v.shape[0] == v_old, (
                f"{k} has leading dim {v.shape[0]}, expected vocab size {v_old}. "
                "Is this checkpoint already extended?"
            )
            out.append(k)
    return out


def transplant_vocab(
    sd: Dict[str, torch.Tensor],
    keys: Sequence[str],
    v_old: int,
    v_new: int,
    seed: int = 0,
    new_row_scale: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Return a new state dict with `keys` resized from v_old to v_new rows.

    Rows [0, v_old) are copied verbatim. Rows [v_old, v_new) get small
    random values scaled relative to the existing table -- nonzero so the
    new symbols are distinguishable, small so they cannot dominate logits.
    """
    assert v_new > v_old, f"v_new ({v_new}) must exceed v_old ({v_old})"
    g = torch.Generator().manual_seed(seed)
    out = dict(sd)

    for k in keys:
        w = sd[k]
        assert w.shape[0] == v_old, f"{k}: leading dim {w.shape[0]} != {v_old}"

        shape = list(w.shape)
        shape[0] = v_new
        new = torch.empty(shape, dtype=w.dtype, device=w.device)

        std = w.float().std().item() * new_row_scale
        new.normal_(0.0, std, generator=g)
        new[:v_old] = w

        assert torch.equal(new[:v_old], w), f"old rows corrupted in {k}"
        out[k] = new

    return out