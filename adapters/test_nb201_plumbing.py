"""Contract / unit smoke tests for NB201 Phase-1 adapter plumbing.

No full GPU matrix, no TSS-15625, no Figshare downloads. Pure ranking /
CLI-gate / genotype helpers; TinyNetwork build is optional when NB201+torch
are importable.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ADAPTERS = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTERS))

import nb201_common as nb  # noqa: E402
import nb201_search_az  # noqa: E402


def test_rank_sum_prefers_higher_scores() -> None:
    infos = [
        {
            "expressivity": 1.0,
            "progressivity": 1.0,
            "trainability": 1.0,
            "complexity": 1.0,
        },
        {
            "expressivity": 10.0,
            "progressivity": 10.0,
            "trainability": 10.0,
            "complexity": 10.0,
        },
    ]
    ranks = nb.compute_az_nas_rank_sum(infos)
    assert ranks[1] > ranks[0]


def test_rank_sum_nonfinite_ranks_worst() -> None:
    infos = [
        nb.non_finite_info(),
        {
            "expressivity": 0.0,
            "progressivity": 0.0,
            "trainability": 0.0,
            "complexity": 1.0,
        },
    ]
    ranks = nb.compute_az_nas_rank_sum(infos)
    assert ranks[1] > ranks[0]
    assert math.isfinite(ranks[0])


def test_genotype_short_stable() -> None:
    g = "|nor_conv_3x3~0|+|skip_connect~0|nor_conv_1x1~1|"
    a = nb.genotype_short(g)
    b = nb.genotype_short(g)
    assert a == b
    assert "_" in a


def test_expand_train_jobs_locked6() -> None:
    ranked = [
        {"genotype": f"g{i}", "genotype_short": f"g{i}", "rank": i}
        for i in range(1, 6)
    ]
    ranked[3]["selection_role"] = "random"
    jobs = nb.expand_train_jobs(ranked)
    assert len(jobs) == 6
    roles = [j["role"] for j in jobs]
    assert roles.count("top-1") == 3
    assert "top-2" in roles and "top-3" in roles and "random" in roles
    top1_seeds = {j["seed"] for j in jobs if j["role"] == "top-1"}
    assert top1_seeds == {0, 1, 2}


def test_refuse_full_tss_without_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        nb201_search_az._refuse_full_if_needed(
            argparse_ns(n_samples=15625, full_tss=False, allow_full_tss=False)
        )
    assert "Refusing full TSS" in str(exc.value)


def test_refuse_full_tss_flag_allows() -> None:
    nb201_search_az._refuse_full_if_needed(
        argparse_ns(n_samples=15625, full_tss=False, allow_full_tss=True)
    )


def argparse_ns(**kwargs):
    class NS:
        pass

    ns = NS()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def test_search_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(ADAPTERS / "nb201_search_az.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--allow_full_tss" in proc.stdout
    assert "--timed_smoke" in proc.stdout


def test_train_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(ADAPTERS / "nb201_train_selected.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "locked6" in proc.stdout
    assert "sgd_nesterov" in proc.stdout or "200" in proc.stdout


def test_train_uses_score_pad_gutenberg_only() -> None:
    assert nb.train_uses_score_pad("gutenberg") is True
    for ds in ("multnist", "cifartile", "geoclassing", "chesseract"):
        assert nb.train_uses_score_pad(ds) is False


@pytest.mark.skipif(
    not (ADAPTERS.parent / "NB201" / "xautodl").is_dir(),
    reason="NB201 tree missing",
)
def test_tiny_network_gutenberg_native_vs_padded() -> None:
    """Native 27×18 crashes residual add; search pad 32×32 forwards cleanly."""
    torch = pytest.importorskip("torch")
    with nb.nb201_context(ADAPTERS.parent):
        g = nb.parse_genotype(
            "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|"
            "+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|"
        )
        net = nb.build_tiny_network(g, num_classes=6, in_channels=1, C=4, N=1)
        with pytest.raises(RuntimeError, match="size of tensor"):
            net(torch.randn(2, 1, 27, 18))
        feat, logits = net(torch.randn(2, 1, 32, 32))
        assert logits.shape == (2, 6)
        assert feat.ndim == 2


@pytest.mark.skipif(
    not (ADAPTERS.parent / "NB201" / "xautodl").is_dir(),
    reason="NB201 tree missing",
)
def test_tiny_network_stem_in_channels() -> None:
    torch = pytest.importorskip("torch")
    with nb.nb201_context(ADAPTERS.parent):
        # Minimal valid genotype: 3 edges for max_nodes=4 structure via str2structure
        g = nb.parse_genotype(
            "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|"
            "+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|"
        )
        for cin in (1, 3, 12):
            net = nb.build_tiny_network(g, num_classes=10, in_channels=cin, C=4, N=1)
            x = torch.randn(2, cin, 16, 16)
            feat, logits = net(x)
            assert logits.shape == (2, 10)
            assert feat.ndim == 2
            feats = net.extract_cell_features(x)
            assert len(feats) >= 2
