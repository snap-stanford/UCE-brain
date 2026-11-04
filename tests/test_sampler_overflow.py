"""Regression test for the cell-sentence overflow guard (CPU only, no model).

A sentence is [CLS] + sample_size genes + two tokens per chromosome, so it can be
longer than pad_length when the sampled genes span many chromosome ids. The
sampler then keeps whole chromosome blocks while they fit, truncates the first
block that does not fit (keeping its end token) and drops the rest. A sentence
that fits is unchanged, and the random draws (hence the prediction targets) are
the same either way. Truncation issues a RuntimeWarning; a fitting sentence does not.

Run with ``uv run python tests/test_sampler_overflow.py`` (pytest also collects
it, if installed).
"""

import warnings

import numpy as np
import torch

from uce_brain.data.sampler import sample_cell_sentences_mapping_gene

CLS, PAD, END, OFFSET = 1, 0, 2, 100_000


def make_cell(n_genes, n_chroms, seed=0):
    """A synthetic cell: most genes expressed, genes spread over ``n_chroms`` chromosome ids."""
    rng = np.random.default_rng(seed)
    counts = torch.from_numpy(rng.integers(0, 20, size=n_genes).astype(np.float32)).unsqueeze(0)
    log_expr = torch.log1p(counts)
    weights = log_expr / log_expr.sum(dim=1, keepdim=True)
    return dict(
        counts=counts,
        batch_weights=weights,
        gene_protein_ids=np.arange(10, 10 + n_genes, dtype=np.int64),
        gene_chroms=rng.integers(0, n_chroms, size=n_genes).astype(np.int64),
        gene_starts=rng.integers(0, 10**8, size=n_genes).astype(np.int64),
    )


def sample(cell, pad_length, sample_size=2048, mask_prop=0.2, seed=7, expect_warning=None):
    # the sampler zeroes masked weights in place, so every call gets fresh copies
    inputs = {k: (v.clone() if torch.is_tensor(v) else v.copy()) for k, v in cell.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sample_cell_sentences_mapping_gene(
            **inputs, pad_length=pad_length, positive_sample_num=64, negative_sample_num=64, mask_prop=mask_prop,
            sample_size=sample_size, cls_token_idx=CLS, chrom_token_offset=OFFSET, chrom_token_right_idx=END,
            pad_token_idx=PAD, seed=seed)
    truncated = [w for w in caught if issubclass(w.category, RuntimeWarning) and "truncated" in str(w.message)]
    if expect_warning is not None:
        assert bool(truncated) == expect_warning, (pad_length, [str(w.message) for w in caught])
    return res


def blocks(sentence):
    """Split an untruncated sentence into [(chrom start token, [gene tokens])]."""
    toks = [int(t) for t in sentence if t != PAD]
    assert toks[0] == CLS
    out, i = [], 1
    while i < len(toks):
        assert toks[i] >= OFFSET, "expected a chromosome start token"
        j = toks.index(END, i)
        out.append((toks[i], toks[i + 1:j]))
        i = j + 1
    return out


def expected_truncation(full_blocks, pad_length):
    out = [CLS]
    for start, genes in full_blocks:
        if len(out) >= pad_length - 2:
            break
        out.append(start)
        out += genes[: pad_length - 1 - len(out)]
        out.append(END)
    return out + [PAD] * (pad_length - len(out))


def real_length(res):
    return int((~res["mask"][0]).sum())


def test_overflow_truncates_and_keeps_rng_stream():
    cell = make_cell(n_genes=3000, n_chroms=3000)
    big = 1 + 2048 + 2 * 3000  # cannot overflow
    full = sample(cell, big, expect_warning=False)
    n_full = real_length(full)
    assert n_full > 3584, n_full  # the 3,584-token context overflows for this cell
    fb = blocks(full["batch_sentences"][0].tolist())
    for pad_length in (3584, 2048, n_full - 1, n_full - 2, n_full - 3):
        res = sample(cell, pad_length, expect_warning=True)
        sent = res["batch_sentences"][0].tolist()
        assert len(sent) == pad_length
        assert sent == expected_truncation(fb, pad_length), pad_length
        n = real_length(res)
        assert n <= pad_length and sent[n - 1] == END and sent[0] == CLS
        assert res["mask"][0, :n].sum() == 0 and res["mask"][0, n:].all()
        # targets are drawn after the sentence: same random stream, same targets
        assert torch.equal(res["cell_outputs_X_pe"], full["cell_outputs_X_pe"])
        assert torch.equal(res["cell_outputs_Y"], full["cell_outputs_Y"])


def test_fitting_sentence_is_unchanged():
    cell = make_cell(n_genes=3000, n_chroms=25)
    full = sample(cell, 4096, expect_warning=False)
    n = real_length(full)
    assert n < 3584
    for pad_length in (n, n + 1, 3584):  # n = exact fit
        res = sample(cell, pad_length, expect_warning=False)
        assert torch.equal(res["batch_sentences"][0, :n], full["batch_sentences"][0, :n]), pad_length
        assert (res["batch_sentences"][0, n:] == PAD).all() and real_length(res) == n
        assert torch.equal(res["cell_outputs_X_pe"], full["cell_outputs_X_pe"])


if __name__ == "__main__":
    test_overflow_truncates_and_keeps_rng_stream()
    test_fitting_sentence_is_unchanged()
    print("ok")
