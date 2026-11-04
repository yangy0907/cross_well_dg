"""Freeze a submission manuscript from v3 without editing earlier candidates."""
from pathlib import Path
import shutil
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse_v3'
DST = ROOT / 'paper/gse_final_20261103'


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for p in SRC.iterdir():
        if p.suffix in {'.tex', '.bib', '.sty', '.cls', '.bst'} or p.name == 'highlights.txt':
            shutil.copy2(p, DST / p.name)
    shutil.copytree(SRC / 'figs', DST / 'figs', dirs_exist_ok=True)
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    supp = (DST / 'supplementary.tex').read_text(encoding='utf-8')
    archived = []
    for label, number in [('tab:ablation2x2', 'S17'), ('tab:paired', 'S18')]:
        # Locate by label; historical paired table's exact label is checked below.
        marker = r'\label{' + label + '}'
        if marker not in text and label == 'tab:paired':
            match = re.search(r'\label\{([^}]+)\}', text[text.index(r'\caption{Historical closed-set feature contrasts'):])
            label = match.group(1); marker = r'\label{' + label + '}'
        at = text.index(marker)
        start = text.rfind(r'\begin{table}', 0, at)
        end = text.index(r'\end{table}', at) + len(r'\end{table}')
        archived.append(text[start:end].replace(r'Table~\ref{tab:v2core}', 'the main-text matched-pilot table'))
        text = text[:start] + text[end:]
        text = text.replace(r'\ref{' + label + '}', number)
    # Remove the mask-confounded legacy row from the two primary means and record.
    lines = text.splitlines()
    removed = []
    for i, line in enumerate(lines):
        if (line.startswith('WellNorm-LGBM (test-domain) & $0.300') or
            line.startswith('WellNorm-LGBM & $0.303') or
            line.startswith('WellNorm & Post / well-')):
            removed.append(line); lines[i] = ''
    if len(removed) != 3:
        raise RuntimeError('Expected three historical rows in primary displays')
    text = '\n'.join(lines) + '\n'
    # Qualify every remaining numeric historical WellNorm table row.
    text = re.sub(r'^(WellNorm(?:-LGBM|-norel)?)(?=\s*(?:\([^\n]*?\))?\s*&)',
                  r'\1 (legacy)', text, flags=re.M)
    text = text.replace('Test-domain & WellNorm-LGBM &', 'Test-domain & WellNorm (legacy) &')
    text = text.replace('Test-domain & WellNorm-norel (no $d^{\\mathrm{rel}}$) &', 'Test-domain & WN-norel (legacy) &')
    # The variant-definition table contains no performance results.
    a = text.index(r'\caption{Method variants')
    b = text.index(r'\end{table}', a)
    segment = text[a:b].replace('WellNorm-norel (legacy)', 'WellNorm-norel').replace('WellNorm-LGBM (legacy)', 'WellNorm-LGBM')
    text = text[:a] + segment + text[b:]
    text = text.replace(r'\subsection{Discussion of mechanism}', r'\subsection{Interpretation and applicability}')
    start = text.index('Protocol choice dominates headline scores')
    end = text.index('\n\n', start)
    text = text[:start] + r'''The results support a conditional interpretation of feature benefit. The primary Global--Hybrid comparison shows a mean improvement under post-drilling information, while the fixed-taxonomy tails and parent-averaged analysis show that this improvement is not uniform across wells. The restricted-curve discovery suggested a larger relative-depth benefit with four curves; the new parent-grouped follow-up reduces the mean difference of benefits to $+0.011$, with only two of three partitions positive. This variation does not establish a universal dependence on curve count, and the simultaneous changes in partition and grouping prevent attributing the difference between phases to parent grouping alone.

The boundary-coordinate experiment provides a more direct within-model check. On identical retained depths, changing only relative depth reduces mean four-curve Macro-F1 across all six fixed crop scenarios, and the mean Hybrid--Hybrid-norel advantage reverses under the bottom and two-sided 20\% stresses. This demonstrates sensitivity to how the observed interval is defined within the evaluated learner. Because full-curve normalisation moments remain frozen, it does not establish performance under physical incomplete logging or identify a geological cause. Relative depth is therefore an interval-dependent post-drilling feature whose benefit should be checked against an explicit logging-support definition.

The historical WellNorm comparisons also lost raw missingness information and cannot isolate the effect of retaining global curve levels. They are retained as implementation diagnostics in Supplementary Tables~S17--S19; corrected comparisons are limited to the matched pilot. The causal-prefix results likewise apply to the observed completed-wireline coverage, not early-bit deployment. These boundaries motivate reporting admissible information, preprocessing support and well-level tails together \citep{Koh2021WILDS,Zhou2023DGSurvey,Gulrajani2021DomainBed}.''' + text[end:]
    start = text.index('On public FORCE~2020 we introduce', text.index(r'\section{Conclusions}'))
    end = text.index(r'\appendix', start)
    text = text[:start] + r'''A fixed LightGBM family on FORCE~2020 shows why cross-well lithofacies scores require an explicit information budget and well-level assessment. Across 18 closed well-held-out repetitions, inductive Macro-F1 is $0.339\pm0.045$, compared with 0.737 under a naive depth split. Post-drilling Hybrid variants reach $0.370$--$0.379$, while improvements in coverage-aware well tails are inconsistent. The optimism contrast combines protocol and domain-composition changes and is not a pure causal leakage estimate.

The restricted-curve pilot and three new parent-grouped partitions show heterogeneous relative-depth gains. In the follow-up, the four-minus-sixteen-curve difference of gains averages $+0.011$ and is positive in two of three partitions. Controlled boundary-coordinate stresses can reverse the mean four-curve Hybrid advantage on identical retained depths. Together, these results support interval- and input-dependent use of the evaluated representations, rather than universal model superiority. Corrected raw-missingness controls qualify the historical WellNorm comparisons; simple well descriptors do not yet supply a validated selection rule.

The contribution is a reproducible framework for linking well-held-out estimation to admissible target information, log availability, interval definitions and coverage-aware tails. The demonstrated models remain interpretation aids. Independent-corpus validation, calibrated rejection and population-level inference remain outstanding.

''' + text[end:]
    text = text.replace('The v2 code and results are local revision artifacts and must be included in the public release before submission.',
                        'The accompanying review package includes the v2 code, results and execution manifests.')
    text = text.replace('for the core six LightGBM methods.}',
        'for the historical core schedule. The WellNorm point is a mask-confounded archival reference, not a corrected-method ranking.}')
    text = text.replace(r'Hybrid$-$HN Worst-fixed can flip relative to the well-id panel (Fig.~\ref{fig:pareto}).}',
        r'Hybrid$-$HN Worst-fixed can flip relative to the well-id panel (Fig.~\ref{fig:pareto}). The WellNorm point is a mask-confounded archival reference, not a corrected-method ranking.}')
    a = text.index(r'\section*{Data availability}')
    b = text.index(r'\section*{Declaration of competing interests}', a)
    availability = r'''\section*{Data availability}
Lithofacies labels and well logs originate from the public FORCE~2020 Machine Learning competition \citep{Bormann2020FORCE} (\href{https://doi.org/10.5281/zenodo.4351156}{10.5281/zenodo.4351156}). The accompanying \texttt{GSE\_review\_materials\_20261103.zip} contains processed input tables, well-split definitions, experiment summaries, per-run results, saved follow-up models and predictions, and source/configuration/data hashes. Its file inventory and SHA-256 manifest distinguish the historical r3 results, corrected four-curve pilot and new parent-grouped follow-up. Historical WellNorm results are identified as mask-confounded diagnostics rather than corrected performance estimates. Raw competition files can be obtained from the cited dataset source. No independent external-corpus result is claimed.

\section*{Code availability}
The CPU-only Python source, locked dependencies, configurations, tests and reproduction instructions for the reported analyses are included in the accompanying review package. The package includes the feature-equivalence checks, reproduction gates and original execution-source snapshots, so historical execution provenance is retained separately from the final packaging inventory. Python~3.11.5 and the versions in \path{requirements.lock.txt} were used for the new experiments. The historical pipeline tag is \texttt{r3\_fullcurve\_stats\_20260930}; the corrected pilot and follow-up have separate v2/v3 manifests. The earlier repository at \url{https://github.com/yangy0907/cross_well_dg} is an additional historical locator and is not asserted to contain these final revision artifacts. The review package is identified by a separate SHA-256 sidecar; no software DOI has been assigned. No public deposition is implied by providing this review package.

'''
    text = text[:a] + availability + text[b:]
    # Explicit, compact archival record for the three removed primary rows.
    archival_rows = r'''\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Historical mask-confounded WellNorm aggregates removed from primary displays. These r3 values are retained for traceability and are not corrected-mask performance estimates or evidence isolating global levels. Corrected WellNorm results appear in the main matched pilot.}
\label{tab:s-legacywnmeans}
\centering\small
\begin{tabular}{@{}llccc@{}}
\toprule
Grouping & Scope & Macro & Worst-fixed & W10-fixed \\
\midrule
Well-id & 18 closed splits & $0.300\pm0.034$ & $0.091\pm0.015$ & $0.104\pm0.015$ \\
Parent-grouped & 18 closed splits & $0.303\pm0.035$ & $0.085\pm0.019$ & $0.098\pm0.018$ \\
\bottomrule
\end{tabular}
\end{table}
'''
    supp = supp.replace(r'\end{document}', '\n\n' + r'\FloatBarrier' + '\n' +
        r'\section*{Historical mask-confounded implementation diagnostics}' + '\n\n' +
        '\n\n'.join(archived) + '\n\n' + archival_rows + '\n' + r'\end{document}')
    # An archived caption refers to the main result spine using an external S number.
    supp = supp.replace(r'\ref{tab:ablation2x2}', 'S17')
    (DST / 'main.tex').write_text(text, encoding='utf-8')
    (DST / 'supplementary.tex').write_text(supp, encoding='utf-8')
    (DST / 'highlights.txt').write_text(
        'Well-held-out scores depend on admissible target-well information.\n'
        'Relative-depth gains vary across log sets and new parent-grouped splits.\n'
        'Boundary-coordinate changes can reverse mean gains on matched scored depths.\n'
        'Fixed-taxonomy tails combine class coverage with predictive performance.\n', encoding='utf-8')
    print('Final sources assembled; earlier manuscript versions unchanged.')


if __name__ == '__main__':
    main()
