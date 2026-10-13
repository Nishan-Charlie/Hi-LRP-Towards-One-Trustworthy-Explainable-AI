"""Compatibility shims for upstream API drift.

Imported for its side effects by ``xai_bench/__init__.py``, before anything
touches LXT.

transformers v5 removed ``transformers.pytorch_utils.find_pruneable_heads_and_indices``,
but LXT's ``lxt/efficient/models/bert.py`` still imports it at module scope, and
that module is imported eagerly by ``lxt.efficient.core``. The result is that
merely importing the LXT backend raises ImportError under transformers>=5, even
though the symbol belongs to a BERT head-pruning path that no vision model ever
reaches.

We restore the symbol with its original v4 implementation rather than a stub, so
that behavior is unchanged in the (unused) event something calls it. Downgrading
transformers globally would be the alternative, but the interpreter is shared
with unrelated projects.
"""
import torch


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Verbatim behavior of the transformers v4 utility of the same name.

    Returns the heads that can still be pruned (excluding already-pruned ones,
    reindexed) and the row indices of the weight matrix to keep.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def apply():
    """Install the shim if the running transformers lacks the symbol."""
    try:
        from transformers import pytorch_utils
    except Exception:
        return False
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
        return True
    return False


applied = apply()
