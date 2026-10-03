"""Apply Major-Revision cuts to main.tex / supplementary.tex (text surgery)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "gse" / "main.tex"
SUPP = ROOT / "paper" / "gse" / "supplementary.tex"


def cut_between(tex: str, start: str, end: str) -> tuple[str, str]:
    """Remove [start, end) markers; return (new_tex, extracted)."""
    i = tex.find(start)
    if i < 0:
        raise SystemExit(f"start not found: {start[:60]!r}")
    j = tex.find(end, i)
    if j < 0:
        raise SystemExit(f"end not found after start: {end[:60]!r}")
    return tex[:i] + tex[j:], tex[i:j]


def replace_once(tex: str, old: str, new: str) -> str:
    if old not in tex:
        raise SystemExit(f"replace target missing: {old[:80]!r}")
    return tex.replace(old, new, 1)


def main() -> None:
    tex = MAIN.read_text(encoding="utf-8")
    moved: list[str] = []

    # --- 1) Sim method paragraph: no main-text Sim scores ---
    tex = replace_once(
        tex,
        r"""rescaled to unit mean and passed to LightGBM as sample weights. The temperature $T$ is fixed in the released configuration; dividing by $\bar{d}$ keeps the effective scale of $T$ roughly invariant to the absolute units of the mean-curve space. Wells whose mean log signature is closer to the target therefore contribute more gradient mass during training. The construction is a lightweight cousin of Chang-style source reweighting \citep{Chang2021GRSL} (Section~\ref{sec:related-crosswell}), using unlabelled $\mu_t$ only. \emph{Support symmetry:} the released code computes both $\mu_w$ and $\mu_t$ on full observable curves (label-independent). The Protocol-1 Sim tables in this manuscript were produced with full-curve $\mu_t$ but source means from labelled training depths only; because well-level labelled fractions range from $\approx$9\% to $\approx$100\%, that asymmetry can mix annotation coverage into the distance. We disclose it here; primary Hybrid versus WellNorm claims do not depend on Sim. Operationally, \emph{per-target-well} similarity (one $\mu_t$ per held-out well) matches one-new-well deployment and is the only Sim mode treated as deployment-relevant: Protocol-1 seeds~42--45 under pipeline~\texttt{r3} are reported in Tables~\ref{tab:p1} and~\ref{tab:simperwell}. Protocols~2--3 retain \emph{batch} $\mu_t$ over all test wells of the active split as offline diagnostics that may overestimate one-well gains. The method must be disclosed as UDA-lite whenever compared with inductive DG.""",
        r"""rescaled to unit mean and passed to LightGBM as sample weights. The temperature $T$ is fixed in the released configuration; dividing by $\bar{d}$ keeps the effective scale of $T$ roughly invariant to the absolute units of the mean-curve space. Wells whose mean log signature is closer to the target therefore contribute more gradient mass during training. The construction is a lightweight cousin of Chang-style source reweighting \citep{Chang2021GRSL} (Section~\ref{sec:related-crosswell}), using unlabelled $\mu_t$ only. The released code computes both $\mu_w$ and $\mu_t$ on full observable curves. Quantitative Hybrid+Sim / Hybrid+Plus scores in this submission were produced with an earlier asymmetric support (full-curve $\mu_t$, labelled-depth $\mu_w$) and are therefore placed in the Supplementary material as \emph{historical UDA-lite diagnostics}, not as main-text results tied to the current code path. Primary Hybrid versus WellNorm claims do not depend on Sim. The method must be disclosed as UDA-lite whenever compared with inductive DG.""",
    )

    tex = replace_once(
        tex,
        r"""We do not prescribe a single production default. Table~\ref{tab:decision} (Section~\ref{sec:multiseed}) only summarises which variants were strongest \emph{under the present tables} for each information setting; it is an observational pattern sheet, not an operations manual. Global-LGBM remains the shared booster for fair hybrid-feature ablations. Global-RF is reported in the main multi-seed table as an inductive reference (unequal $\leq$120k-row schedule). For one-new-well UDA-lite, the matched setting is \emph{per-target-well} similarity (Protocol-1 Tables~\ref{tab:p1} and~\ref{tab:simperwell}); quadrant / LOWO Hybrid+Sim / Hybrid+Plus scores remain \emph{batch} diagnostics.""",
        r"""We do not prescribe a single production default. Global-LGBM remains the shared booster for fair hybrid-feature ablations. Global-RF is reported in the main multi-seed table as an inductive reference (unequal $\leq$120k-row schedule). UDA-lite Hybrid+Sim / Plus scores are Supplementary historical diagnostics only.""",
    )

    # --- 2) Remove UDA block from tab:p1 + Sim paragraph ---
    tex = replace_once(
        tex,
        r"""For UDA-lite, the deployment-matched setting is \emph{per-target-well} Hybrid+Sim: on seed~42 it reaches Macro-F1 0.414 / worst-well 0.195, close to \emph{batch} Sim (0.410 / 0.221). Across all four seeds the per-well schedule yields mean Macro-F1 0.375 / worst-well 0.191 (Table~\ref{tab:simperwell}). Source-only MLP reaches 0.365 / 0.111; CORAL-MLP stays appendix-only.

""",
        r"""Source-only MLP reaches 0.365 / 0.111; CORAL-MLP and historical Hybrid+Sim / Plus scores stay in the appendix / Supplementary material.

""",
    )
    tex = replace_once(
        tex,
        r"""Test-domain & Hybrid+Smooth & 0.415 & 0.221 \\
\midrule
\multicolumn{4}{@{}l@{}}{\emph{UDA-lite (full-curve $\mu_t$ may enter training)}} \\
UDA-lite & Hybrid+Sim (per-well; deployment) & 0.414 & 0.195 \\
UDA-lite & Hybrid+Sim (batch; offline diagnostic) & 0.410 & 0.221 \\
UDA-lite & Hybrid+Plus (batch; offline diagnostic) & 0.406 & 0.211 \\
\bottomrule""",
        r"""Test-domain & Hybrid+Smooth & 0.415 & 0.221 \\
\bottomrule""",
    )

    # Drop seed-42 while detail table; keep whilems
    tex = replace_once(
        tex,
        r"""Whole-well $z$-scores and relative depth assume a completed logging interval. To make that operational constraint explicit, Table~\ref{tab:while} contrasts four LightGBM variants on the same Protocol-1 seed~42 split: inductive Global-LGBM; post-drilling Hybrid (whole-well $z$ + relative depth); Hybrid-norel (whole-well $z$, no relative depth); and Hybrid-while (causal expanding prefix $z$ along the \emph{full} curve up to the current depth, no relative depth). On this split Hybrid-while raises mean Macro-F1 to 0.433 and worst-well to 0.205 (versus Global 0.404 / 0.144). Cold-start coverage and prefix-length bins are reported in Supplementary Table~S1 (script \texttt{while\_coldstart\_study.py}).

Table~\ref{tab:whilems} repeats Global versus Hybrid-while over Protocol-1 seeds. The four-seed mean Macro-F1 is 0.379 versus 0.359 and mean worst-well 0.210 versus 0.177 (worst-well higher on 3/4 under r3; seed~45 is the exception).

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-1 seed~42: inductive vs post-drilling vs while-drilling Hybrid features (pipeline r3). Causal prefix uses full-curve depths.}
\label{tab:while}
\begin{tabular*}{\tblwidth}{@{}llccc@{}}
\toprule
Setting & Method & Well-$z$ & Macro-F1 & Worst-well F1 \\
\midrule
Inductive DG & Global-LGBM & --- & 0.404 & 0.144 \\
Test-domain (post) & Hybrid & whole-well + rel.\ depth & 0.411 & 0.217 \\
Test-domain (post) & Hybrid-norel & whole-well, no rel.\ depth & 0.404 & 0.238 \\
Test-domain (while) & Hybrid-while & causal prefix, no rel.\ depth & 0.433 & 0.205 \\
\bottomrule
\end{tabular*}
\end{table}

""",
        r"""Whole-well $z$-scores and relative depth assume a completed logging interval. Hybrid-while replaces whole-well moments by a causal expanding prefix along the full curve (no relative depth). Table~\ref{tab:whilems} summarises Global versus Hybrid-while over Protocol-1 seeds: four-seed mean Macro-F1 0.379 versus 0.359 and mean worst-well 0.210 versus 0.177 (higher on 3/4 seeds; seed~45 is the exception). Cold-start bins are in Supplementary Table~S1.

""",
    )

    # Extract simperwell table to supp, delete from main
    tex, chunk = cut_between(
        tex,
        r"""\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-1 Hybrid+Sim:""",
        r"""\subsection{Multi-seed robustness}\label{sec:multiseed}""",
    )
    moved.append("% --- historical Sim (asymmetric source means) ---\n" + chunk)

    # Multiseed text + drop perseed/openset/decision; keep checklist/paired/multiseed slim
    tex = replace_once(
        tex,
        r"""Protocol-1 multi-seed summaries use the four \emph{pre-specified} seeds~42--45 under pipeline~\texttt{r3} (full-curve well aggregates; Section~\ref{sec:claims}). Seed~44 is open-set (Table~\ref{tab:regime}) but is retained in the primary mean; we additionally report a closed-set companion mean over 42/43/45. Table~\ref{tab:perseed} lists every seed. The \emph{primary} paired contrast is Hybrid versus WellNorm (same information budget): mean Macro-F1 lift $+0.092$ (bootstrap 95\% CI $[0.051,0.134]$) and W10 lift $+0.042$ (CI $[0.015,0.069]$); worst-well $\Delta$ is positive on 3/4 seeds with bootstrap CI that includes zero (Table~\ref{tab:paired}). Hybrid versus inductive Global-LGBM is reported only as a descriptive cross-budget contrast (worst-well wins 2/4 under r3). Equal-row Global-LGBM ($\leq$120k) and Global-RF both exceed full-split Global-LGBM on mean Macro-F1 (Table~\ref{tab:multiseed}), so learner/row-budget choice remains important beside feature design.

Table~\ref{tab:checklist} turns the information-budget taxonomy into a reusable reporting checklist. Table~\ref{tab:decision} remains a short navigation aid.
""",
        r"""Protocol-1 multi-seed summaries use the four \emph{pre-specified} seeds~42--45 under pipeline~\texttt{r3}. Seed~44 is open-set (Table~\ref{tab:regime}) but is retained in the primary mean. The \emph{primary} paired contrast is Hybrid versus WellNorm: mean Macro-F1 lift $+0.092$ (seed-level bootstrap 95\% CI $[0.051,0.134]$) and W10 lift $+0.042$ (CI $[0.015,0.069]$); worst-well $\Delta$ is positive on 3/4 seeds with CI that includes zero (Table~\ref{tab:paired}). Hybrid versus inductive Global-LGBM is a descriptive cross-budget contrast (worst-well wins 2/4). Equal-row Global-LGBM ($\leq$120k) and Global-RF both exceed full-split Global-LGBM on mean Macro-F1 (Table~\ref{tab:multiseed}). Per-seed and open-set detail tables are in the Supplementary material. Table~\ref{tab:checklist} is the reusable reporting checklist.
""",
    )

    tex, chunk = cut_between(
        tex,
        r"""\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-1 per-seed Global-LGBM vs Hybrid vs WellNorm""",
        r"""\begin{table}[width=.95\linewidth,pos=htbp]
\caption{Protocol-1 multi-seed summary""",
    )
    moved.append("% --- per-seed detail ---\n" + chunk)

    tex = replace_once(
        tex,
        r"""Hybrid-while (test-domain) & 0.379 & 0.083 & 0.210 & $+0.020$ \\
Hybrid+Sim (per-well; deployment) & 0.375 & 0.067 & 0.191 & $+0.016$ \\
Hybrid+Sim (batch; diagnostic) & 0.373 & 0.068 & 0.195 & $+0.014$ \\
Hybrid+Plus (batch; diagnostic) & 0.357 & 0.058 & 0.178 & $-0.003$ \\
\bottomrule""",
        r"""Hybrid-while (test-domain) & 0.379 & 0.083 & 0.210 & $+0.020$ \\
\bottomrule""",
    )

    tex, chunk = cut_between(
        tex,
        r"""\begin{table}[width=.95\linewidth,pos=htbp]
\caption{Seed~44 open-set diagnostics""",
        r"""\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Reporting checklist for FORCE~2020""",
    )
    moved.append("% --- open-set detail ---\n" + chunk)

    tex, chunk = cut_between(
        tex,
        r"""\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Navigation aid for the present tables""",
        r"""\subsection{Protocol-2: quadrant geographic stress test (illustrative)}\label{sec:p2}""",
    )
    # delete decision table entirely (do not move)
    del chunk

    # --- Embargo: shorten + move table ---
    tex, chunk = cut_between(
        tex,
        r"""\subsection{Fixed-test-block depth embargo}\label{sec:embargo}

To probe adjacent-depth structure under a \emph{fixed} test-block design, we hold out contiguous depth blocks inside Protocol-1 seed~42 test wells ($\approx$25\% of each well) and allow remaining labelled depths from those wells into training only outside an embargo of $\{0,1,5,10,20\}$\,m from the block boundary (\texttt{scripts/embargo\_fixed\_block\_study.py}; pipeline~\texttt{r3}). Training row count is matched to the source-only reference by repeated subsampling (ten seeds). Mean Macro-F1 stays near $0.42$ across embargos ($0.422\pm0.006$ at $0$\,m to $0.416\pm0.005$ at $20$\,m), while mean worst-well Macro-F1 falls from $0.085$ ($0$\,m) to $0.041$ ($20$\,m) (Table~\ref{tab:embargo}). Class-prior TV to the test block is essentially flat ($\approx$0.149), so the worst-well drop is consistent with removing near-boundary neighbour structure rather than a prior-shift confound. Together with Table~\ref{tab:matched}, this supports reading the 0.737$\to$0.36 gap as a mixed protocol/optimism effect, not a pure leakage causal estimate.

\begin{table}[width=.95\linewidth,pos=htbp]
\caption{Fixed-test-block embargo""",
        r"""\subsection{Related wellbores and parent-grouped sensitivity}\label{sec:parent}""",
    )
    # Keep short pointer; move full table to supp
    embargo_table = chunk[chunk.find("\\begin{table}") :]
    moved.append("% --- embargo grid ---\n" + embargo_table)
    tex = (
        tex[: tex.find(r"\subsection{Related wellbores")]
        + r"""\subsection{Fixed-test-block depth embargo}\label{sec:embargo}

A fixed-test-block embargo inside Protocol-1 seed~42 test wells (pipeline~\texttt{r3}; ten equal-$n$ subsample seeds) keeps mean Macro-F1 near $0.42$ while worst-well Macro-F1 falls from $0.085$ ($0$\,m) to $0.041$ ($20$\,m); class-prior TV stays flat ($\approx$0.149). Full grid: Supplementary Table~\ref{tab:embargo}. Together with Table~\ref{tab:matched}, the 0.737$\to$0.36 gap is a mixed protocol/optimism effect, not a pure leakage causal estimate.

"""
        + tex[tex.find(r"\subsection{Related wellbores") :]
    )

    # --- Parent: shorten, move table (also r2-tagged) ---
    tex, chunk = cut_between(
        tex,
        r"""\subsection{Related wellbores and parent-grouped sensitivity}\label{sec:parent}

Protocol-1 partitions by wellbore identity.""",
        r"""\begin{figure}[width=.75\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig1_leakage_vs_crosswell.png}""",
    )
    parent_table = chunk[chunk.find("\\begin{table}") :] if "\\begin{table}" in chunk else ""
    if parent_table:
        moved.append("% --- parent-well (historical/sensitivity) ---\n" + parent_table)
    tex = (
        tex[: tex.find(r"\begin{figure}[width=.75\linewidth,pos=htbp]")]
        + r"""\subsection{Related wellbores and parent-grouped sensitivity}\label{sec:parent}

Protocol-1 splits by wellbore identity. Among seeds~42--45 only seed~44 places related NPD wellbores of parent \texttt{34/5-1} across train/test ($\approx$1.33\,km surface separation). A parent-grouped sensitivity and drop-sibling ablation (Supplementary Table~\ref{tab:parent}) leave the four-seed Global-LGBM mean near 0.36, so the overlap is disclosed for integrity but does not drive that seed's headline score.

"""
        + tex[tex.find(r"\begin{figure}[width=.75\linewidth,pos=htbp]") :]
    )

    # --- Protocol-2: drop Sim rows + figure ---
    tex = replace_once(
        tex,
        r"""Protocol-2 is an illustrative geographic stress test (Section~\ref{sec:claims}): two NPD \emph{quadrants} (leading FORCE name token) and one partition seed; Hybrid+Sim / Hybrid+Plus use batch $\mu_t$ as offline diagnostics. Table~\ref{tab:p2} and Fig.~\ref{fig:p2} report quadrants~15 and~35 under pipeline~\texttt{r3} (seed~42), blocked by information budget. On quadrant~15, batch Hybrid+Plus reaches 0.414 Macro-F1 against 0.344 for WellNorm and 0.308 for inductive Global-LGBM (cross-budget description); Hybrid-while is 0.400. On the harder quadrant~35, scores remain lower; Hybrid+Smooth attains 0.335 among the LightGBM variants reported here. These lifts illustrate that geographic hold-out can remain stringent; they must not be extrapolated beyond the reported quadrants, must not be read as NPD-block hold-out, and batch Sim must not be read as one-new-well deployment. Global-RF under the unequal schedule is reported in Appendix~\ref{app:rf}.

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-2 Macro-F1 for held-out NPD quadrants (pipeline r3; seed~42 only; two quadrants; illustrative). Blocked by information budget; batch Sim / Plus are offline diagnostics. Not an NPD-block hold-out.}
\label{tab:p2}
\begin{tabular*}{\tblwidth}{@{}llcc@{}}
\toprule
Setting & Method & Quadrant 15 & Quadrant 35 \\
\midrule
Inductive DG & Global-LGBM & 0.308 & 0.272 \\
Inductive DG & Global-LGBM-eq ($\leq$120k) & 0.381 & 0.312 \\
Inductive DG & Global-RF ($\leq$120k) & 0.375 & 0.283 \\
Inductive DG & MLP (source-only) & 0.320 & 0.301 \\
\midrule
Test-domain & WellNorm-LGBM & 0.344 & 0.285 \\
Test-domain & Hybrid & 0.346 & 0.330 \\
Test-domain & Hybrid+Smooth & 0.346 & 0.335 \\
Test-domain & Hybrid-while & 0.400 & 0.328 \\
\midrule
UDA-lite (batch; diagnostic) & Hybrid+Sim & 0.349 & 0.312 \\
UDA-lite (batch; diagnostic) & Hybrid+Plus & 0.414 & 0.325 \\
\bottomrule
\end{tabular*}
\end{table}

\begin{figure}[width=.9\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig4_protocol2_block_holdout.png}
\caption{Protocol-2 Macro-F1 for NPD quadrants~15 and~35 (seed~42 only; illustrative stress test; not block hold-out).}
\label{fig:p2}
\end{figure}
""",
        r"""Protocol-2 is an illustrative geographic stress test: two NPD \emph{quadrants} and one seed under pipeline~\texttt{r3} (Table~\ref{tab:p2}). On quadrant~15, Hybrid-while reaches 0.400 versus WellNorm 0.344 and inductive Global-LGBM 0.308; on harder quadrant~35, Hybrid+Smooth attains 0.335. These lifts must not be extrapolated beyond the reported quadrants or read as NPD-block hold-out. Historical batch Sim / Plus and the Protocol-2 bar figure are in the Supplementary material.

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-2 Macro-F1 for held-out NPD quadrants (pipeline r3; seed~42; illustrative). Not an NPD-block hold-out.}
\label{tab:p2}
\begin{tabular*}{\tblwidth}{@{}llcc@{}}
\toprule
Setting & Method & Quadrant 15 & Quadrant 35 \\
\midrule
Inductive DG & Global-LGBM & 0.308 & 0.272 \\
Inductive DG & Global-LGBM-eq ($\leq$120k) & 0.381 & 0.312 \\
Inductive DG & Global-RF ($\leq$120k) & 0.375 & 0.283 \\
\midrule
Test-domain & WellNorm-LGBM & 0.344 & 0.285 \\
Test-domain & Hybrid & 0.346 & 0.330 \\
Test-domain & Hybrid+Smooth & 0.346 & 0.335 \\
Test-domain & Hybrid-while & 0.400 & 0.328 \\
\bottomrule
\end{tabular*}
\end{table}
""",
    )

    # --- Protocol-3: drop r2 alt + Sim rows; keep primary ---
    tex = replace_once(
        tex,
        r"""Protocol-3 is a fixed-draw stress test rather than a population LOWO over all 118 labelled wells (Section~\ref{sec:claims}); Hybrid+Sim / Hybrid+Plus again use batch $\mu_t$. Six primary targets were completed under pipeline~\texttt{r3} for inductive Global-LGBM, WellNorm and the hybrid family (selection in Section~\ref{sec:problem}). Sample std over wells ($N-1$) is used throughout. Mean Macro-F1 rises from $0.381\pm0.110$ (WellNorm) to $0.419\pm0.096$ for Hybrid and $0.418\pm0.094$ for Hybrid+Smooth (Table~\ref{tab:p3}; Fig.~\ref{fig:p3}). Inductive Global-LGBM reaches $0.360\pm0.071$---below WellNorm on this draw---so the Hybrid LOWO gain is not an artefact of a weak WellNorm baseline alone. UDA-lite add-ons raise the six-well average modestly further (Hybrid+Plus $0.429\pm0.084$). Per-well behaviour remains uneven (Fig.~\ref{fig:p3well}).

An alternate spaced six-well set (\texttt{selection=spaced\_alt}: \{16/1-6~A, 16/5-3, 25/10-9, 31/4-10, 34/8-7~R, 35/9-7\}) was previously reported under pipeline~\texttt{r2} (Table~\ref{tab:p3alt}; Fig.~\ref{fig:lowosel}); the r3 re-run of that alternate draw is pending and the r2 numbers are retained only for selection-sensitivity context. Absolute means remain selection-dependent: on the alternate r2 draw, Hybrid raises the mean from $0.362\pm0.087$ (WellNorm) to $0.370\pm0.093$, while Hybrid+Plus reaches $0.406\pm0.096$; inductive Global-LGBM is $0.342\pm0.155$. The primary-versus-alternate discrepancy is therefore best read as \emph{selection sensitivity of the six-well mean}, not as evidence that Hybrid is uniformly stable across the full well population---which is why we report both draws and caution against treating either mean as a field-wide guarantee. Unequal-budget Global-RF LOWO numbers are deferred to Appendix~\ref{app:rf}.
""",
        r"""Protocol-3 is a fixed-draw stress test on six primary targets under pipeline~\texttt{r3} (selection in Section~\ref{sec:problem}), not a population LOWO over 118 wells. Mean Macro-F1 rises from $0.381\pm0.110$ (WellNorm) to $0.419\pm0.096$ for Hybrid and $0.418\pm0.094$ for Hybrid+Smooth (Table~\ref{tab:p3}; Fig.~\ref{fig:p3}). Inductive Global-LGBM reaches $0.360\pm0.071$---below WellNorm on this draw. Per-well behaviour remains uneven (Fig.~\ref{fig:p3well}). Historical UDA-lite LOWO scores and alternate spaced draws are omitted from the main text to keep a single r3 pipeline.
""",
    )
    tex = replace_once(
        tex,
        r"""Test-domain & Hybrid+Smooth & $0.418\pm0.094$ & +0.037 \\
\midrule
UDA-lite & Hybrid+Sim (batch) & $0.424\pm0.099$ & +0.043 \\
UDA-lite & Hybrid+Plus (batch) & $0.429\pm0.084$ & +0.048 \\
\bottomrule""",
        r"""Test-domain & Hybrid+Smooth & $0.418\pm0.094$ & +0.037 \\
\bottomrule""",
    )
    tex = replace_once(
        tex,
        r"""\caption{Protocol-3 LOWO Macro-F1 over six target wells (mean$\pm$sample std; pipeline r3; illustrative fixed draw---\emph{not} a population estimate over 118 wells). Hybrid+Sim / Plus use \emph{batch} similarity. Global-RF omitted (Appendix~\ref{app:rf}).}""",
        r"""\caption{Protocol-3 LOWO Macro-F1 over six target wells (mean$\pm$sample std; pipeline r3; illustrative fixed draw---\emph{not} a population estimate over 118 wells).}""",
    )

    # Delete r2 alt table + figure
    tex, _ = cut_between(
        tex,
        r"""\begin{table}[width=.95\linewidth,pos=htbp]
\caption{Protocol-3 LOWO selection sensitivity:""",
        r"""\subsection{Hard-well case notes under LOWO}""",
    )

    # Hard-well: remove Sim number
    tex = replace_once(
        tex,
        r"""WellNorm already reaches Macro-F1 0.566. Hybrid methods raise the score to about 0.59--0.62 (Hybrid+Sim 0.616; Hybrid+Smooth 0.603). Here the source pool apparently contains wells with sufficiently similar log signatures, so both hybrid features and similarity weighting help.""",
        r"""WellNorm already reaches Macro-F1 0.566. Hybrid methods raise the score to about 0.59--0.62 (Hybrid+Smooth 0.603). Here the source pool apparently contains wells with sufficiently similar log signatures.""",
    )

    # --- Delete fixed-taxonomy ---
    tex = replace_once(
        tex,
        r"""\subsection{Per-class behaviour, fixed-taxonomy check and sensitivity}\label{sec:rare}

Macro-F1 averages hide strong class imbalance (Fig.~\ref{fig:class}). Fig.~\ref{fig:perclass} compares per-class F1 for Global and Hybrid under Protocol-1 seed~42. Dominant classes (Shale, Sandstone, Halite) already score relatively high; the hybrid stack mainly lifts Sandstone and Sandstone/Shale, while Dolomite and Chalk remain near zero F1 and Coal/Tuff stay weak. Confusion analysis of rare classes (Fig.~\ref{fig:rarecm}) shows that Chalk is mostly absorbed into Limestone, Dolomite into Shale/Sandstone, and Tuff into Shale ($\approx$62--65\% of true Tuff depths). Absolute cross-well Macro-F1 near 0.40 is therefore limited as much by rare-class sink errors as by average domain shift.

\paragraph{Fixed-taxonomy companion (historical diagnostic).}
An earlier check re-scored per-well Macro-F1 over the full training label set for every well, scoring absent classes as F1~$=0$ (\texttt{zero\_division=0}). Absolute scores then fall largely because wells lack many facies, so the denominator is dominated by structural zeros rather than composition-adjusted accuracy. We therefore \emph{do not} treat fixed-taxonomy as evidence that composition bias has been removed. Table~\ref{tab:fixedtax} is retained only as a denominator diagnostic under pipeline~\texttt{r3}; under \texttt{well\_true}, Hybrid versus Global worst-well wins are 2/4 (mean $\Delta\approx+0.003$), matching the descriptive cross-budget tally elsewhere. Primary risk reporting remains \texttt{well\_true} worst-well and worst-10\%, together with per-class F1 (Fig.~\ref{fig:perclass}).

\paragraph{Relative-depth ablation.}
On seed~42 (pipeline r3), disabling relative depth inside the hybrid stack (keeping global curves, well-wise $z$-scores and masks) yields Macro-F1 0.404 / worst-well 0.238 (\texttt{well\_true}), versus full Hybrid 0.411 / 0.217 (Table~\ref{tab:p1}). Together with the weak WellNorm-only row in Table~\ref{tab:p1}, this indicates that the Hybrid lift over Global is driven mainly by \emph{concatenating} global levels with well-wise $z$-scores; $d^{\mathrm{rel}}$ is an optional post-drilling convenience feature whose contribution should be validated per deployment rather than assumed as part of the Hybrid ``sell''.

Seed~44 well-level Hybrid$-$Global gaps correlate more with class-prior total-variation distance to the training wells ($r\approx-0.31$) than with mean curve missingness ($r\approx0.08$; Supplementary Fig.~S5; $N=17$ test wells). This association figure remains a pipeline~\texttt{r2} diagnostic and is not used for r3 headline claims. These are exploratory associations rather than a causal mechanism proof. The defaults $T=1.0$ and smoothing window~7 were fixed in the released configuration \emph{before} the Protocol-1 seed~42 sensitivity curves in Supplementary Figs.~S6--S7; those curves score the held-out test partition of that seed, were \emph{not} used for model selection, and are reported only as post-hoc diagnostics. Nevertheless, because this project iterated over several pipeline revisions while inspecting Protocol-1 seed~42 summaries, \emph{implicit} researcher degrees of freedom cannot be ruled out from disclosure alone. Future releases should move temperature/window selection to well-held-out validation (or nested well-level CV) and reserve an untouched final well partition.

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Protocol-1 fixed-taxonomy check (pipeline r3). Top: seed~42 detail (\texttt{well\_true} vs fixed; Hybrid-no-reldepth disables relative depth). Bottom: seeds~42--45 worst-well win counts and mean paired $\Delta$ (Hybrid $-$ Global; wins after three-decimal rounding).}
\label{tab:fixedtax}
\begin{tabular*}{\tblwidth}{@{}lcccc@{}}
\toprule
Method (seed~42) & Worst (\texttt{well\_true}) & Worst (fixed) & Well-mean (\texttt{well\_true}) & Well-mean (fixed) \\
\midrule
Global-LGBM & 0.144 & 0.079 & 0.378 & 0.222 \\
Hybrid & 0.217 & 0.118 & 0.432 & 0.252 \\
Hybrid-no-reldepth & 0.238 & 0.130 & 0.422 & 0.244 \\
\midrule
\multicolumn{5}{l}{Seeds~42--45: Hybrid$>$Global worst-well wins / mean $\Delta$} \\
\texttt{well\_true} & \multicolumn{2}{c}{2/4 / $+0.003$} & \multicolumn{2}{c}{---} \\
fixed-taxonomy & \multicolumn{2}{c}{2/4 / $+0.008$} & \multicolumn{2}{c}{---} \\
\bottomrule
\end{tabular*}
\end{table}
""",
        r"""\subsection{Per-class behaviour and sensitivity}\label{sec:rare}

Macro-F1 averages hide strong class imbalance (Fig.~\ref{fig:class}). Fig.~\ref{fig:perclass} compares per-class F1 for Global and Hybrid under Protocol-1 seed~42. Dominant classes (Shale, Sandstone, Halite) already score relatively high; the hybrid stack mainly lifts Sandstone and Sandstone/Shale, while Dolomite and Chalk remain near zero F1 and Coal/Tuff stay weak. Confusion analysis of rare classes (Fig.~\ref{fig:rarecm}) shows that Chalk is mostly absorbed into Limestone, Dolomite into Shale/Sandstone, and Tuff into Shale ($\approx$62--65\% of true Tuff depths). Absolute cross-well Macro-F1 near 0.40 is therefore limited as much by rare-class sink errors as by average domain shift.

On seed~42 (pipeline r3), disabling relative depth inside the hybrid stack yields Macro-F1 0.404 / worst-well 0.238, versus full Hybrid 0.411 / 0.217 (Table~\ref{tab:p1}). Together with the weak WellNorm-only row, this is \emph{consistent with} the Hybrid lift arising mainly from concatenating global levels with well-wise $z$-scores; $d^{\mathrm{rel}}$ remains optional. Defaults $T=1.0$ and smoothing window~7 were fixed before the Supplementary sensitivity curves on seed~42's test partition (post-hoc diagnostics only).
""",
    )

    # Soften metrics fixed-taxonomy mention + forcepen move pointer
    tex = replace_once(
        tex,
        r"""We additionally inspect per-class F1 and well-level Hybrid$-$Global paired gaps. An earlier \emph{fixed-taxonomy} companion (averaging over the full training label set, with absent classes scored as F1~$=0$) is reported only as a historical diagnostic in Section~\ref{sec:rare}: because missing classes dominate the denominator, it does \emph{not} cleanly remove composition bias. Well-level standard deviations use sample std ($N-1$). Accuracy and weighted F1 are secondary because the label distribution is strongly skewed (Fig.~\ref{fig:class}; \citealp{Japkowicz2002Imbalance}). Headline scores use Macro-F1 rather than the FORCE competition penalty matrix; an exploratory FORCE-penalty companion under Protocol-1 is reported in Section~\ref{sec:forcecmp} \citep{Bormann2020FORCE}.""",
        r"""We additionally inspect per-class F1 (Section~\ref{sec:rare}). Well-level standard deviations use sample std ($N-1$). Accuracy and weighted F1 are secondary because the label distribution is strongly skewed (Fig.~\ref{fig:class}; \citealp{Japkowicz2002Imbalance}). Headline scores use Macro-F1; an exploratory FORCE-penalty companion is in the Supplementary material \citep{Bormann2020FORCE}.""",
    )

    # Claim-scope shorten
    tex = replace_once(
        tex,
        r"""\item Protocol-1 seeds~42--45 are \emph{pre-specified exploratory} partitions (consecutive integers fixed before headline aggregation; not a public time-stamped preregistration). Seed~42 additionally hosts ablations and Supplementary sensitivity curves on that seed's test partition. All four seeds enter the primary multi-seed mean; seed~44 is tagged \emph{open-set} (4.4\% unseen test labels) and is reported with known-class-only Macro-F1 / worst-well / W10 alongside count diagnostics (Tables~\ref{tab:regime}--\ref{tab:openset}), not dropped post hoc. Protocol-1 splits by wellbore identity: among seeds~42--45 only seed~44 places related NPD wellbores of the same parent across train and test (\texttt{34/5-1 S} in train, \texttt{34/5-1 A} in test; median Euclidean surface distance from FORCE $x_{\mathrm{loc}}$/$y_{\mathrm{loc}}$ $\approx$1.33\,km; Section~\ref{sec:parent}). A parent-grouped sensitivity and a drop-sibling ablation accompany the primary tables.
\item Multi-seed summaries use $n=4$ (42--45). Worst-well ``win counts'' are descriptive sign tallies. The primary paired table (Table~\ref{tab:paired}) reports Hybrid$-$WellNorm $\Delta$ with \emph{seed-level} bootstrap CIs (resampling the four seed deltas only; not a hierarchical well/block bootstrap); with $n=4$ the CI width should not be over-interpreted. Hybrid$-$Global ranges remain descriptive cross-budget min--max. Matched equal-$n$ leak draws ($D_{\mathrm{eq}}$) and depth-embargo grids use a fixed test-block seed and repeated subsample seeds (Sections~\ref{sec:matched}--\ref{sec:embargo}).
\item Protocols~2 and~3 remain geographic / LOWO stress tests (two NPD quadrants; two fixed six-well draws).
\item Main-text inductive baselines include Global-LGBM and Global-RF (RF uses a $\leq$120k-row schedule; equal-budget RF is noted as a limit). CORAL-MLP remains an appendix unequal-budget reference.
\item UDA-lite: per-target-well Sim is the deployment-matched mode (Protocol-1; pipeline~\texttt{r3}). Batch Sim / Plus remain offline diagnostics.
\item Absolute Macro-F1 near a four-seed mean of $\approx$0.36 supports interpreter decision support (Section~\ref{sec:ops}), not stand-alone automation. Confirmed Hybrid claims emphasise Macro-F1 / W10 under the same target-information budget versus WellNorm; worst-well is reported directionally because its seed-level CI includes zero.""",
        r"""\item Seeds~42--45 are \emph{pre-specified exploratory} partitions ($n=4$). Seed~44 is open-set (and the sole related-wellbore collision; Section~\ref{sec:parent}) and is retained with known-class diagnostics in the Supplementary material, not dropped post hoc.
\item Paired Hybrid$-$WellNorm contrasts use \emph{seed-level} bootstrap CIs (four seed deltas only; not hierarchical). Worst-well win counts are descriptive; the worst-well CI includes zero.
\item Protocols~2--3 are fixed-draw geographic / LOWO stress tests. Historical Hybrid+Sim / Plus scores are Supplementary only (asymmetric source means). CORAL-MLP remains appendix-only; Global-RF uses an unequal $\leq$120k-row schedule.
\item Absolute Macro-F1 near $\approx$0.36 supports interpreter decision support (Section~\ref{sec:ops}), not stand-alone automation.""",
    )

    # Discussion soften
    tex = replace_once(
        tex,
        r"""Three observations follow from the tables and from Section~\ref{sec:claims}. First, protocol choice dominates headline scores: depth-point Macro-F1 of 0.737 and closed-set well-hold-out means near 0.37 describe different evaluation problems (optimism gap, not a pure leakage causal estimate). Second, under well-held-out scoring the Hybrid lift on worst-well risk is explained primarily by concatenating global curve levels with well-wise $z$-scores (WellNorm alone is weaker; removing relative depth does not remove the lift; Section~\ref{sec:rare}); Hybrid-while retains a worst-well lift on closed-set seeds without full-well moments (Tables~\ref{tab:while}--\ref{tab:whilems}). Same-budget claims prefer Hybrid versus WellNorm; Global contrasts are cross-budget and descriptive. Mean Macro-F1 edges remain secondary. Third, component stacking is not monotone, so Table~\ref{tab:decision} is a navigation sheet. Method claims must state the information budget. These points align with DG evaluation practice that emphasises robustness under distribution shift \citep{Koh2021WILDS,Zhou2023DGSurvey,Gulrajani2021DomainBed}.""",
        r"""Three observations follow. First, protocol choice dominates headline scores: depth-point Macro-F1 of 0.737 and well-hold-out means near 0.36 describe different evaluation problems (optimism gap, not a pure leakage causal estimate). Second, the Macro-F1 and W10 gains over WellNorm are consistent with improved reliability-aware transfer; the worst-well improvement remains directional because its confidence interval includes zero. Feature ablations are consistent with concatenating global levels and well-wise $z$-scores as the main Hybrid ingredient (Section~\ref{sec:rare}); Hybrid-while retains a directional worst-well lift without full-well moments (Table~\ref{tab:whilems}). Method claims must state the information budget. These points align with DG evaluation practice under distribution shift \citep{Koh2021WILDS,Zhou2023DGSurvey,Gulrajani2021DomainBed}.""",
    )

    # Ops: drop decision table refs
    tex = replace_once(
        tex,
        r"""In deployment the protocols translate into an acceptance loop that respects the information budgets and the pattern sheet in Table~\ref{tab:decision}. When a newly drilled well arrives with unlabelled curves, a model trained only on historical wells should be judged under a well-hold-out mindset: mean Macro-F1 on labelled offsets---if any---is secondary to worst-interval risk. Global-LGBM is the inductive DG option when target-well aggregates must not be used; Hybrid-while is the matched option when prefix statistics are admissible but total depth / full-well moments are not; Hybrid (global + whole-well $z$; relative depth optional) suits post-drilling whole-well interpretation; Hybrid+Plus / Hybrid+Sim may reweight source wells only when unlabelled target logs may enter training, and then preferably with \emph{per-target-well} similarity. Centred majority smoothing is an offline post-processor, not a while-drilling filter.""",
        r"""In deployment, a model trained only on historical wells should be judged under a well-hold-out mindset: mean Macro-F1 is secondary to worst-interval risk. Global-LGBM is the inductive option when target aggregates must not be used; Hybrid-while when only prefix statistics are admissible; Hybrid for post-drilling whole-well interpretation. UDA-lite reweighting requires disclosing unlabelled-target use and should be re-run under the released symmetric full-curve support before operational claims. Centred majority smoothing is an offline post-processor.""",
    )

    # Move FORCE penalty paragraph+table to supp (keep FORCE discussion subsection)
    tex, chunk = cut_between(
        tex,
        r"""\paragraph{FORCE penalty companion (exploratory).}
For readers oriented to the competition leaderboard, Table~\ref{tab:forcepen}""",
        r"""Our UDA-lite reference remains deliberately weak""",
    )
    moved.append(
        r"""\section*{FORCE penalty companion (historical diagnostic)}

For readers oriented to the competition leaderboard, Table~\ref{tab:forcepen}"""
        + chunk
    )
    tex = replace_once(
        tex,
        r"""Our UDA-lite reference remains deliberately weak""",
        r"""An exploratory FORCE-penalty companion under Protocol-1 is tabulated in the Supplementary material. Our UDA-lite reference remains deliberately weak""",
    )
    tex = tex.replace(
        r"""Within that envelope, Hybrid features raise worst-well Macro-F1 on closed-set exploratory seeds as a cross-budget descriptive contrast; mean Macro-F1 edges remain secondary.""",
        r"""Within that envelope, Hybrid versus WellNorm gains on Macro-F1 / W10 are the primary same-budget contrast; worst-well remains directional (CI includes zero).""",
    )

    # Pre-r2 -> Pre-r3
    tex = tex.replace("Pre-r2 historical", "Pre-r3 historical")
    tex = tex.replace("archive\\_pre\\_r2/", "archive\\_pre\\_r3/")
    tex = tex.replace("archive_pre_r2/", "archive_pre_r3/")

    # Abstract: drop embargo mention density
    tex = replace_once(
        tex,
        r"""Matched within-well depth-block, fixed-test-well leakage and depth-embargo runs (Sections~\ref{sec:matched}--\ref{sec:embargo}) show that adjacent-depth structure and well/class shift both matter; we do not treat 0.737$-$0.36 as a pure leakage penalty.""",
        r"""Matched leakage contrasts (Section~\ref{sec:matched}) show that adjacent-depth structure and well/class shift both matter; we do not treat 0.737$-$0.36 as a pure leakage penalty.""",
    )

    # Appendix refs to while / simperwell
    tex = tex.replace(
        r"""Protocol-1 reports per-target-well Sim for seeds~42--45 (Tables~\ref{tab:p1} and~\ref{tab:simperwell}) alongside batch diagnostics.""",
        r"""Historical Protocol-1 Sim scores are in the Supplementary material.""",
    )
    tex = tex.replace(
        r"""Hybrid-while (Tables~\ref{tab:while}--\ref{tab:whilems}; worst-well 3/4 vs Global)""",
        r"""Hybrid-while (Table~\ref{tab:whilems}; worst-well 3/4 vs Global)""",
    )

    # Data coverage sentence about sim tables
    tex = tex.replace(
        r"""similarity means $\mu_t$ (and, in the released code, source $\mu_w$) likewise use full-curve support, while the Sim tables reported here used labelled-depth source means (Section~\ref{sec:method}). """,
        r"""similarity means use full-curve support in the released code (historical Sim scores: Supplementary material). """,
    )

    MAIN.write_text(tex, encoding="utf-8")
    print("wrote", MAIN)

    # --- Supplementary updates ---
    supp = SUPP.read_text(encoding="utf-8")
    # Remove r2 association figure (S5) from supp
    if r"fig_followup_seed44_mechanism.png" in supp:
        supp, _ = cut_between(
            supp,
            r"""\begin{figure}[width=.9\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig_followup_seed44_mechanism.png}""",
            r"""\FloatBarrier

\section*{Hyper-parameter sensitivity (post-hoc diagnostics)}""",
        )
        # restore FloatBarrier + section header
        idx = supp.find(r"\section*{Hyper-parameter sensitivity")
        if idx < 0:
            # cut removed up to section - need to re-add
            pass
        # cut_between left content starting at Hyper-parameter - check
        if not supp.rstrip().endswith("diagnostics)}"):
            # After cut, next should be Hyper-parameter section
            if r"\section*{Hyper-parameter sensitivity" not in supp:
                raise SystemExit("supp cut failed for S5")

    # Fix abstract
    supp = replace_once(
        supp,
        r"""Additional figures for the main text: class-prior and feature drift under Protocol-1, hybrid ablations, seed-44 diagnostics, and post-hoc (non-nested) hyper-parameter sensitivity curves for similarity temperature and depth-smoothing windows. Multi-seed fixed-taxonomy and FORCE penalty companions are tabulated in the main text (\texttt{results/followup/}). See the main-text Claim scope section for how these panels relate to confirmatory claims.""",
        r"""Supplementary tables and figures for the main text. Hybrid+Sim / Plus scores below are \emph{historical} (full-curve $\mu_t$ with labelled-depth source $\mu_w$) and do not match the released symmetric full-curve code path. Pipeline~\texttt{r2} association panels are omitted. See the main-text Claim scope for confirmatory status.""",
    )

    # Ensure table numbering S*
    if r"\renewcommand{\thetable}{S\arabic{table}}" not in supp:
        supp = replace_once(
            supp,
            r"""\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{figure}{0}""",
            r"""\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{figure}{0}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{0}""",
        )

    # Append moved blocks before \end{document}
    appendix = "\n\n\\FloatBarrier\n\\section*{Moved from main text (revision compression)}\n\n"
    appendix += "\\textbf{Note.} Hybrid+Sim / Plus tables are historical UDA-lite diagnostics (asymmetric source means).\n\n"
    appendix += "\n\n".join(moved)
    # Protocol-2 figure
    appendix += r"""

\begin{figure}[width=.9\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig4_protocol2_block_holdout.png}
\caption{Protocol-2 Macro-F1 for NPD quadrants~15 and~35 (seed~42; pipeline r3; illustrative).}
\label{fig:p2}
\end{figure}
"""
    if "\\end{document}" not in supp:
        raise SystemExit("no end{document} in supp")
    supp = supp.replace("\\end{document}", appendix + "\n\\end{document}", 1)
    SUPP.write_text(supp, encoding="utf-8")
    print("wrote", SUPP)
    print("moved chunks:", len(moved))


if __name__ == "__main__":
    main()
