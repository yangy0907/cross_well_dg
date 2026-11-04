"""Cross-well split protocols."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import BLOCK_COL, SPLITS_DIR, WELL_COL


def parent_well_name(well: str) -> str:
    """NPD-style parent well: strip trailing trajectory tokens (A, S, R, T4, ...).

    Example: ``34/5-1 A`` / ``34/5-1 S`` → ``34/5-1``. In the labelled FORCE
    corpus used here only one parent has multiple wellbores under this rule.
    """
    import re

    return re.sub(r"\s+[A-Za-z0-9]+$", "", str(well).strip())


def parent_train_test_collisions(
    train_wells: list[str], test_wells: list[str]
) -> list[dict[str, Any]]:
    """Parents that appear in both train and test (related-wellbore leakage)."""
    from collections import defaultdict

    by_parent: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    for w in train_wells:
        by_parent[parent_well_name(w)]["train"].append(str(w))
    for w in test_wells:
        by_parent[parent_well_name(w)]["test"].append(str(w))
    out = []
    for parent, sides in sorted(by_parent.items()):
        if sides["train"] and sides["test"]:
            out.append(
                {
                    "parent": parent,
                    "train_wellbores": sorted(sides["train"]),
                    "test_wellbores": sorted(sides["test"]),
                }
            )
    return out


def _save_split(name: str, payload: dict[str, Any]) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def protocol1_random_wells(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Protocol-1: random wellbore-id split (no depth leakage).

    Does **not** group NPD parent wells; related wellbores (e.g. ``34/5-1 S`` /
    ``34/5-1 A``) may land across train and test. See
    :func:`protocol1_parent_grouped_wells` and ``parent_train_test_collisions``.

    Set ``save=False`` in unit tests so toy wells cannot overwrite
    ``data/splits/protocol1_seed*.json``.
    """
    wells = np.array(sorted(df[WELL_COL].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(wells)
    n = len(wells)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = wells[:n_train].tolist()
    val = wells[n_train : n_train + n_val].tolist()
    test = wells[n_train + n_val :].tolist()
    payload = {
        "protocol": "protocol1_random_wells",
        "seed": seed,
        "train_wells": train,
        "val_wells": val,
        "test_wells": test,
        "parent_train_test_collisions": parent_train_test_collisions(train, test),
    }
    if save:
        _save_split(f"protocol1_seed{seed}", payload)
    return payload


def protocol1_parent_grouped_wells(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Protocol-1 sensitivity: assign all wellbores of a parent to the same split.

    Fractions are applied over unique :func:`parent_well_name` keys, then expanded
    to wellbore IDs. Prevents train/test leakage between e.g. ``34/5-1 S`` and
    ``34/5-1 A``.
    """
    from collections import defaultdict

    wells = sorted(df[WELL_COL].astype(str).unique())
    by_parent: dict[str, list[str]] = defaultdict(list)
    for w in wells:
        by_parent[parent_well_name(w)].append(w)
    parents = np.array(sorted(by_parent.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(parents)
    n = len(parents)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_p = parents[:n_train].tolist()
    val_p = parents[n_train : n_train + n_val].tolist()
    test_p = parents[n_train + n_val :].tolist()

    def expand(ps: list[str]) -> list[str]:
        out: list[str] = []
        for p in ps:
            out.extend(sorted(by_parent[p]))
        return out

    train = expand(train_p)
    val = expand(val_p)
    test = expand(test_p)
    payload = {
        "protocol": "protocol1_parent_grouped_wells",
        "seed": seed,
        "train_wells": train,
        "val_wells": val,
        "test_wells": test,
        "n_parents_train": len(train_p),
        "n_parents_val": len(val_p),
        "n_parents_test": len(test_p),
        "parent_train_test_collisions": parent_train_test_collisions(train, test),
    }
    if save:
        _save_split(f"protocol1_parent_grouped_seed{seed}", payload)
    return payload


def protocol2_quadrant_holdout(
    df: pd.DataFrame,
    holdout_blocks: list[str],
    val_frac_of_source: float = 0.15,
    seed: int = 42,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """Protocol-2: hold out one or more NPD quadrants as unseen target domain.

    ``holdout_blocks`` keeps its historical argument/CSV name; values are NPD
    quadrant tokens (e.g. ``15``, ``35``), not ``15/9``-style QUADRANT/BLOCK
    ids. The protocol key remains ``protocol2_block_holdout`` so existing
    result CSVs resume correctly.
    """
    holdout = {str(b) for b in holdout_blocks}
    well_block = df.groupby(WELL_COL)[BLOCK_COL].first().astype(str)
    test_wells = well_block[well_block.isin(holdout)].index.tolist()
    source_wells = well_block[~well_block.isin(holdout)].index.tolist()
    if not test_wells:
        raise ValueError(f"No wells in holdout quadrants {holdout_blocks}")
    if not source_wells:
        raise ValueError("No source wells left after quadrant holdout")

    rng = np.random.default_rng(seed)
    src = np.array(sorted(source_wells))
    rng.shuffle(src)
    n_val = max(1, int(round(len(src) * val_frac_of_source)))
    val = src[:n_val].tolist()
    train = src[n_val:].tolist()
    tag = "-".join(sorted(holdout))
    payload = {
        "protocol": "protocol2_block_holdout",
        "seed": seed,
        "holdout_blocks": sorted(holdout),
        "train_wells": train,
        "val_wells": val,
        "test_wells": sorted(test_wells),
    }
    if save:
        _save_split(f"protocol2_holdout_{tag}_seed{seed}", payload)
    return payload


# Backward-compatible alias (historical name; values are quadrants).
protocol2_block_holdout = protocol2_quadrant_holdout


def _protocol3_wells_hash(all_wells: list[str]) -> str:
    import hashlib

    payload = ("|".join(map(str, all_wells))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _protocol3_selection_params_hash(
    *,
    max_targets: int,
    selection: str,
    seed: int,
    target_wells: list[str],
) -> str:
    import hashlib
    import json as _json

    payload = _json.dumps(
        {
            "max_targets": int(max_targets),
            "selection": str(selection),
            "seed": int(seed),
            "target_wells": list(map(str, target_wells)),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def protocol3_leave_one_well_out(
    df: pd.DataFrame,
    target_wells: list[str] | None = None,
    max_targets: int = 12,
    seed: int = 42,
    selection: str = "spaced",
    *,
    save: bool = True,
) -> list[dict[str, Any]]:
    """Protocol-3: LOWO on a subset of wells (CPU-friendly).

    selection:
      - spaced: index-spaced over wells sorted by (block, n)
      - spaced_alt: same sort, indices shifted to form a disjoint-ish sensitivity set
      - random: RNG sample of max_targets wells

    Cached split JSON is reused only when ``wells_hash`` and
    ``selection_params_hash`` match the current dataset / selection args.
    Mismatch (or legacy cache missing those fields) raises rather than
    silently reusing a stale split.
    """
    import hashlib

    all_wells = sorted(map(str, df[WELL_COL].unique()))
    wells_hash = _protocol3_wells_hash(all_wells)
    if target_wells is None:
        summary = (
            df.groupby(WELL_COL)
            .agg(block=(BLOCK_COL, "first"), n=("depth", "size"))
            .reset_index()
        )
        summary = summary.sort_values(["block", "n"]).reset_index(drop=True)
        n_pick = min(max_targets, len(summary))
        if selection == "random":
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(summary), size=n_pick, replace=False)
            target_wells = summary.iloc[sorted(pick)][WELL_COL].tolist()
        else:
            idx = np.linspace(0, len(summary) - 1, num=n_pick, dtype=int)
            if selection == "spaced_alt":
                shift = max(1, len(summary) // (2 * max(1, n_pick)))
                idx = (idx + shift) % len(summary)
                # unique while preserving order
                seen = set()
                uniq = []
                for i in idx:
                    if int(i) not in seen:
                        seen.add(int(i))
                        uniq.append(int(i))
                # fill from unused if collision reduced count
                for j in range(len(summary)):
                    if len(uniq) >= n_pick:
                        break
                    if j not in seen:
                        seen.add(j)
                        uniq.append(j)
                idx = np.asarray(uniq[:n_pick], dtype=int)
            target_wells = summary.iloc[idx][WELL_COL].tolist()
    else:
        target_wells = list(map(str, target_wells))

    selection_params_hash = _protocol3_selection_params_hash(
        max_targets=max_targets,
        selection=selection,
        seed=seed,
        target_wells=target_wells,
    )

    splits = []
    for tw in target_wells:
        safe = tw.replace("/", "_").replace(" ", "")
        # primary spaced keeps original filenames for resume compatibility
        if selection == "spaced":
            existing = SPLITS_DIR / f"protocol3_{safe}_seed{seed}.json"
        else:
            existing = SPLITS_DIR / f"protocol3_{selection}_{safe}_seed{seed}.json"
        if existing.exists():
            payload = json.loads(existing.read_text(encoding="utf-8"))
            cached_wells = str(payload.get("wells_hash", ""))
            cached_sel = str(payload.get("selection_params_hash", ""))
            same_target = str(payload.get("target_well", "")) == str(tw)
            same_sel = str(payload.get("selection") or "spaced") == str(selection)
            same_seed = int(payload.get("seed", -1)) == int(seed)
            if (
                cached_wells == wells_hash
                and cached_sel == selection_params_hash
                and same_target
                and same_sel
                and same_seed
            ):
                splits.append(payload)
                continue
            raise RuntimeError(
                f"Protocol-3 split cache stale or incomplete: {existing.name}. "
                f"Delete it (or regenerate splits) rather than silently reusing. "
                f"wells_hash ok={cached_wells == wells_hash}; "
                f"selection_params_hash ok={cached_sel == selection_params_hash}; "
                f"target/selection/seed ok={same_target and same_sel and same_seed}"
            )

        others = [w for w in all_wells if w != tw]
        tw_digest = int(hashlib.md5(tw.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + (tw_digest % 10000))
        rng.shuffle(others)
        n_val = max(1, int(round(0.15 * len(others))))
        val = others[:n_val]
        train = others[n_val:]
        payload = {
            "protocol": "protocol3_lowo",
            "seed": seed,
            "target_well": tw,
            "selection": selection,
            "max_targets": int(max_targets),
            "wells_hash": wells_hash,
            "selection_params_hash": selection_params_hash,
            "train_wells": train,
            "val_wells": val,
            "test_wells": [tw],
        }
        if save:
            if selection == "spaced":
                _save_split(f"protocol3_{safe}_seed{seed}", payload)
            else:
                _save_split(f"protocol3_{selection}_{safe}_seed{seed}", payload)
        splits.append(payload)
    return splits


def apply_split(df: pd.DataFrame, split: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tr = df[df[WELL_COL].isin(split["train_wells"])].copy()
    va = df[df[WELL_COL].isin(split["val_wells"])].copy()
    te = df[df[WELL_COL].isin(split["test_wells"])].copy()
    return {"train": tr, "val": va, "test": te}


def naive_depth_split(
    df: pd.DataFrame, seed: int = 42, test_frac: float = 0.15, val_frac: float = 0.15
) -> dict[str, Any]:
    """Negative control: random depth-point split (leaky, for paper contrast).

    Returns train/val/test frames plus identity fields for provenance hashing
    (``protocol``, ``seed``, fractions, ``row_index_hash``).
    """
    import hashlib

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    te_idx = np.sort(idx[:n_test])
    va_idx = np.sort(idx[n_test : n_test + n_val])
    tr_idx = np.sort(idx[n_test + n_val :])
    te = df.iloc[te_idx].copy()
    va = df.iloc[va_idx].copy()
    tr = df.iloc[tr_idx].copy()
    # Stable fingerprint of the shuffled partition (position indices into df).
    payload = (
        f"naive|{seed}|{test_frac}|{val_frac}|"
        f"tr={','.join(map(str, tr_idx.tolist()))}|"
        f"va={','.join(map(str, va_idx.tolist()))}|"
        f"te={','.join(map(str, te_idx.tolist()))}"
    ).encode("utf-8")
    row_index_hash = hashlib.sha256(payload).hexdigest()
    return {
        "train": tr,
        "val": va,
        "test": te,
        "protocol": "naive_depth_split",
        "seed": int(seed),
        "test_frac": float(test_frac),
        "val_frac": float(val_frac),
        "row_index_hash": row_index_hash,
    }


def contiguous_depth_block_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Within-well contrast: hold out contiguous depth blocks per well.

    Same wells appear in train/val/test (no well-identity shift). Relative to
    :func:`naive_depth_split` this reduces point-wise adjacency leakage, but
    test depths / class priors need not match exactly—treat as a protocol
    sensitivity contrast, not a pure causal isolation of neighbour leakage.
    """
    rng = np.random.default_rng(seed)
    depth_col = "depth" if "depth" in df.columns else df.columns[0]
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, g in df.groupby(WELL_COL, sort=False):
        g = g.sort_values(depth_col).reset_index(drop=True)
        n = len(g)
        if n < 8:
            # tiny wells: put all in train to keep test well-set stable
            train_parts.append(g)
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_val = max(1, n - n_test - 1)
        # place a contiguous test block, then a contiguous val block in remainder
        max_start = n - n_test
        t0 = int(rng.integers(0, max_start + 1))
        test_idx = set(range(t0, t0 + n_test))
        remain = [i for i in range(n) if i not in test_idx]
        # contiguous val: pick a start in remain as a consecutive run when possible
        if len(remain) <= n_val:
            val_idx = set(remain[:-1] if len(remain) > 1 else remain)
        else:
            # find a contiguous span inside remain of length n_val
            placed = False
            # try random starts along remain sequence
            for _ in range(32):
                s = int(rng.integers(0, len(remain) - n_val + 1))
                cand = remain[s : s + n_val]
                if cand[-1] - cand[0] + 1 == n_val and list(range(cand[0], cand[0] + n_val)) == cand:
                    val_idx = set(cand)
                    placed = True
                    break
            if not placed:
                val_idx = set(remain[:n_val])
        train_idx = [i for i in range(n) if i not in test_idx and i not in val_idx]
        train_parts.append(g.iloc[train_idx])
        val_parts.append(g.iloc[sorted(val_idx)])
        test_parts.append(g.iloc[sorted(test_idx)])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    import hashlib

    h = hashlib.sha256()
    h.update(f"block|{seed}|{test_frac}|{val_frac}|".encode("utf-8"))
    for tag, frame in (("tr", train), ("va", val), ("te", test)):
        h.update(tag.encode("utf-8"))
        wells = frame[WELL_COL].astype(str).to_numpy()
        depths = frame[depth_col].astype(float).to_numpy()
        for w, d in zip(wells, depths):
            h.update(f"{w}:{d};".encode("utf-8"))
    row_index_hash = h.hexdigest()
    return {
        "train": train,
        "val": val,
        "test": test,
        "protocol": "contiguous_depth_block_split",
        "seed": int(seed),
        "test_frac": float(test_frac),
        "val_frac": float(val_frac),
        "row_index_hash": row_index_hash,
    }


def fixed_test_wells_depth_leak_split(
    df: pd.DataFrame,
    test_wells: list[str],
    seed: int = 42,
    test_frac_within: float = 0.5,
    train_wells: list[str] | None = None,
    val_wells: list[str] | None = None,
    val_frac_of_source: float = 0.15,
    match_n_train: int | None = None,
    subsample_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fixed source/test well identity; optionally allow target-well depth leak.

    When ``train_wells`` / ``val_wells`` are supplied (recommended for paired
    contrasts with Protocol-1), the *only* deliberate difference versus a
    matched no-leak control is whether non-test depths from ``test_wells``
    enter training. Source composition and validation wells stay fixed.

    If those lists are omitted, source wells are split at random (legacy path;
    not a single-factor pair with Protocol-1).

    ``match_n_train``: if set and training has more rows, downsample without
    replacement so training size matches a no-leak control.

    ``subsample_seed``: RNG for downsampling only (defaults to ``seed``). Keep
    ``seed`` fixed to freeze contiguous test-block placement while repeating
    equal-n draws.
    """
    rng = np.random.default_rng(seed)
    rng_sub = np.random.default_rng(seed if subsample_seed is None else subsample_seed)
    test_set = set(map(str, test_wells))
    target = df[df[WELL_COL].astype(str).isin(test_set)].copy()
    depth_col = "depth" if "depth" in df.columns else "depth_md"

    if train_wells is not None and val_wells is not None:
        train_set = set(map(str, train_wells))
        val_set = set(map(str, val_wells))
        overlap = (train_set | val_set) & test_set
        if overlap:
            raise ValueError(f"train/val wells overlap test wells: {sorted(overlap)[:5]}")
        train_src = df[df[WELL_COL].astype(str).isin(train_set)].copy()
        val_src = df[df[WELL_COL].astype(str).isin(val_set)].copy()
    else:
        # Legacy: re-draw source train/val (confounds paired C*/D contrasts).
        source = df[~df[WELL_COL].astype(str).isin(test_set)].copy()
        src_wells = np.array(sorted(source[WELL_COL].astype(str).unique()))
        rng.shuffle(src_wells)
        n_val = max(1, int(round(len(src_wells) * val_frac_of_source)))
        val_set = set(src_wells[:n_val].tolist())
        train_src = source[~source[WELL_COL].astype(str).isin(val_set)].copy()
        val_src = source[source[WELL_COL].astype(str).isin(val_set)].copy()

    leak_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, g in target.groupby(WELL_COL, sort=False):
        g = g.sort_values(depth_col).reset_index(drop=True)
        n = len(g)
        n_te = max(1, int(round(n * test_frac_within)))
        if n_te >= n:
            n_te = max(1, n - 1)
        max_start = n - n_te
        t0 = int(rng.integers(0, max_start + 1))
        te = g.iloc[t0 : t0 + n_te]
        leak = g.drop(te.index)
        leak_parts.append(leak)
        test_parts.append(te)

    train = pd.concat([train_src, *leak_parts], ignore_index=True)
    if match_n_train is not None and len(train) > match_n_train:
        take = rng_sub.choice(len(train), size=int(match_n_train), replace=False)
        train = train.iloc[np.sort(take)].reset_index(drop=True)
    val = val_src
    test = pd.concat(test_parts, ignore_index=True)
    import hashlib

    train_wells_out = sorted(train[WELL_COL].astype(str).unique().tolist())
    val_wells_out = sorted(val[WELL_COL].astype(str).unique().tolist())
    test_wells_out = sorted(test[WELL_COL].astype(str).unique().tolist())

    h = hashlib.sha256()
    h.update(
        f"leak|{seed}|{test_frac_within}|{match_n_train}|{subsample_seed}|".encode("utf-8")
    )
    h.update(f"trw={','.join(train_wells_out)}|vaw={','.join(val_wells_out)}|".encode())
    h.update(f"tew={','.join(test_wells_out)}|".encode())
    for tag, frame in (("tr", train), ("va", val), ("te", test)):
        h.update(tag.encode("utf-8"))
        wells = frame[WELL_COL].astype(str).to_numpy()
        depths = frame[depth_col].astype(float).to_numpy()
        for w, d in zip(wells, depths):
            h.update(f"{w}:{d};".encode("utf-8"))
    row_index_hash = h.hexdigest()
    return {
        "train": train,
        "val": val,
        "test": test,
        "protocol": "fixed_test_wells_depth_leak_split",
        "seed": int(seed),
        "test_frac": float(test_frac_within),
        "val_frac": float(val_frac_of_source),
        "train_wells": train_wells_out,
        "val_wells": val_wells_out,
        "test_wells": test_wells_out,
        "row_index_hash": row_index_hash,
        "match_n_train": match_n_train,
        "subsample_seed": int(seed if subsample_seed is None else subsample_seed),
    }
