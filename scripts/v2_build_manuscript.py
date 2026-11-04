"""Build a separate, evidence-bounded v2 manuscript from the frozen v1 candidate.

Requires completed analyses; never edits the original paper/gse or candidate backup.
"""
from pathlib import Path
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse'
DST = ROOT / 'paper/gse_v2'
PILOT = ROOT / 'results/v2/pilot_common_core'
ANALYSIS = ROOT / 'results/v2/existing_well_analysis'


def replace_paragraph(text, start, new):
    pos = text.index(start)
    end = text.index('\n\n', pos)
    return text[:pos] + new + text[end:]


def table(headers, rows, columns):
    lines = [r'\begin{tabular}{@{}' + columns + r'@{}}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    lines.extend(' & '.join(str(v) for v in row) + r' \\' for row in rows)
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}'])


def main():
    notes = json.loads((PILOT / 'summary_notes.json').read_text(encoding='utf-8'))
    control = json.loads((ROOT / 'results/v2/mask_thread_control/control_report.json').read_text(encoding='utf-8'))
    if notes['new_fit_count'] != 20:
        raise RuntimeError('Incomplete pilot')
    DST.mkdir(parents=True, exist_ok=True)
    for name in ['main.tex', 'supplementary.tex', 'refs.bib', 'supplementary-refs.bib',
                 'cas-common.sty', 'cas-sc.cls', 'cas-model2-names.bst']:
        shutil.copy2(SRC / name, DST / name)
    shutil.copytree(SRC / 'figs', DST / 'figs', dirs_exist_ok=True)
    for folder, name in [(PILOT, 'fig_v2_common_core'),
                         (ANALYSIS, 'fig_v2_well_heterogeneity'),
                         (ANALYSIS, 'fig_v2_shift_decomposition')]:
        for ext in ['pdf', 'png']:
            shutil.copy2(folder / f'{name}.{ext}', DST / 'figs' / f'{name}.{ext}')
    shutil.copy2(PILOT / 'table_v2_common_core.tex', DST / 'table_v2_common_core.tex')

    text = (DST / 'main.tex').read_text(encoding='utf-8')
    text = text.replace(r'\newcommand{\revmark}[1]{\textcolor{red}{#1}}',
                        r'\newcommand{\revmark}[1]{#1}')
    text = text.replace('Information-budget-aware validation of cross-well lithofacies classifiers under domain shift: a well-held-out FORCE~2020 benchmark',
                        'Cross-well lithofacies classification under restricted information and log availability: a well-held-out FORCE~2020 study')
    # Remove causal attribution that the mask audit no longer supports.
    text = replace_paragraph(text, 'Hybrid-family behaviour is read from a $2\\times2$ ablation',
        r'Hybrid-family effects are evaluated jointly with the feature and missingness budget. The historical $2\times2$ table is retained for traceability, but WellNorm variants in pipeline r3 lost their raw missingness indicators after normalisation. Historical Hybrid--WellNorm gaps therefore do not isolate global levels. Section~\ref{sec:v2pilot} adds a bounded matched-row four- versus sixteen-curve pilot with raw missingness indicators preserved for every newly fitted method. Comparisons with inductive Global also cross target-information budgets.')
    old = r'We report the full $2\times2$ over \{well-$z$, well-$z$+$d^{\mathrm{rel}}$, global+well-$z$, global+well-$z$+$d^{\mathrm{rel}}$\} (WellNorm-norel / WellNorm / Hybrid-norel / Hybrid). Single-factor global-level contrasts hold $d^{\mathrm{rel}}$ fixed; Hybrid-norel$-$WellNorm alone is a two-factor observational gap and is not used for attribution.'
    new = r'The historical $2\times2$ over \{well-$z$, well-$z$+$d^{\mathrm{rel}}$, global+well-$z$, global+well-$z$+$d^{\mathrm{rel}}$\} (WellNorm-norel / WellNorm / Hybrid-norel / Hybrid) is a record of pipeline r3, not a clean factorial attribution: its WellNorm branches lost raw missingness indicators. The new matched pilot preserves these indicators (Section~\ref{sec:v2design}). Hybrid--Hybrid-norel still isolates adding relative depth within each curve set, subject to the same training schedule.'
    if old not in text:
        raise RuntimeError('Hybrid paragraph changed')
    text = text.replace(old, new)
    text = replace_paragraph(text, r'\revmark{Primary closed multi-rep means, paired changes and Pareto panels are stated',
        r'Primary closed multi-rep means, paired changes and Pareto panels are stated in Section~\ref{sec:primary}. This subsection retains historical three-seed tables for traceability. The nominal global-level gaps, Hybrid$-$WellNorm ($+0.104$ Macro) and Hybrid-norel$-$WellNorm-norel ($+0.093$), also change the availability of raw missingness indicators in pipeline r3. They are therefore descriptive gaps, not isolated global-level effects. The corrected limited-scope comparison is Section~\ref{sec:v2pilot}; it does not replace all 18 historical WellNorm repetitions. Per-seed and open-set detail is in the Supplementary material.')
    text = text.replace('Single-factor global-level contrasts hold $d^{\\mathrm{rel}}$ fixed (right column minus left). The two-factor Hybrid-norel$-$WellNorm gap is not a single-factor attribution.',
        'Nominal global-level contrasts also change raw missingness indicators in pipeline r3 and cannot isolate a global-level effect. Values are archival; corrected pilot results appear in Table~\\ref{tab:v2core}.')
    text = text.replace('Single-factor $\\Delta$ (add global)', 'Nominal $\\Delta$ (mask-confounded)')
    text = text.replace('Closed-set single-factor and two-factor contrasts', 'Historical closed-set feature contrasts with a missingness-mask confound')
    text = text.replace('Hybrid-norel$-$WellNorm changes both global features and relative depth; prefer the single-factor rows for attribution.',
        'All contrasts involving legacy WellNorm also change raw missingness indicators; none isolates global levels. The labels below describe nominal feature switches only.')
    text = text.replace('Hybrid-norel$-$WellNorm-norel (single-factor)', 'Hybrid-norel$-$WellNorm-norel (nominal)')
    text = text.replace('Hybrid$-$WellNorm (single-factor)', 'Hybrid$-$WellNorm (nominal)')
    old = 'Read with the closed-set $2\\times2$ (Table~\\ref{tab:ablation2x2}), the single-factor global-level contrasts are Hybrid-norel$-$WellNorm-norel and Hybrid$-$WellNorm; the two-factor Hybrid-norel$-$WellNorm gap is not attributed to global levels alone.'
    text = text.replace(old, 'The historical contrasts involving WellNorm are confounded by missingness-mask loss and are retained only as an implementation record; the corrected pilot is Table~\\ref{tab:v2core}.')
    text = text.replace('The $2\\times2$ ablation supports reading global levels and relative depth jointly; single-factor global-level lifts are of order $+0.09$--$0.10$ Macro-F1 when $d^{\\mathrm{rel}}$ is held fixed.',
        'The historical $2\\times2$ ablation cannot identify the contribution of global levels because its WellNorm branches also lost raw missingness indicators. The corrected limited pilot and the parent-averaged analysis instead ask how log availability changes the observed feature benefit and whether simple well descriptors explain its heterogeneity (Sections~\\ref{sec:v2pilot}--\\ref{sec:v2heterogeneity}). These analyses do not establish a causal geological or instrument mechanism.')
    text = text.replace('Same-budget Hybrid-family claims emphasise Hybrid-norel versus WellNorm, not superiority over RF on mean Macro-F1.',
        'Historical Hybrid--WellNorm gaps are confounded by missingness-mask loss; the corrected bounded pilot is reported separately. These companions do not establish algorithm superiority.')
    text = text.replace('Prefer Hybrid-norel / Hybrid for post-drilling whole-well interpretation after reading the Pareto / $2\\times2$ ablation',
        'Consider Hybrid-norel / Hybrid for post-drilling whole-well interpretation after checking available curves, the corrected pilot and the Pareto panels')
    # Warn at each table presenting legacy WellNorm numbers.
    text = text.replace(r'\subsection{Primary closed multi-rep results}',
        r'\input{v2_methods}' + '\n\n' + r'\subsection{Primary closed multi-rep results}', 1)
    text = text.replace(r'\subsection{Protocol-2 and Protocol-3 (illustrative stress tests)}',
        r'\input{v2_results}' + '\n\n' + r'\subsection{Protocol-2 and Protocol-3 (illustrative stress tests)}', 1)
    historical_notice = (r'\paragraph{Legacy WellNorm qualification.} All pipeline-r3 WellNorm and WellNorm-norel values in this section and the historical Protocol-2/3 tables are reproduced implementation results. Their missingness indicators were computed after $z$-score replacement of missing values by zero. They must not be used to infer that well normalisation intrinsically harms classification or to isolate the value of retaining global levels. Global and Hybrid features are unchanged by the raw-mask correction; the 18-split Global/Hybrid headline comparison remains applicable. Corrected WellNorm evidence is limited to the new pilot (Section~\ref{sec:v2pilot}).')
    text = text.replace(r'\subsection{Primary closed multi-rep results}\label{sec:primary}',
        r'\subsection{Primary closed multi-rep results}\label{sec:primary}' + '\n\n' + historical_notice)
    text = text.replace('All runs used CPU-only Python~3.11', 'Historical pipeline-r3 runs used CPU-only Python~3.11')
    text = text.replace('Reported rows use \\texttt{pipeline\\_version=r3\\_fullcurve\\_stats\\_20260930}.',
        r'Historical rows use \texttt{pipeline\_version=r3\_fullcurve\_stats\_20260930}; the new restricted-curve and raw-mask rows use \texttt{v2\_matched\_curves\_rawmask\_20261103} and separate provenance under \path{results/v2/}. The provisional public code locator above describes the earlier release; the v2 source, data/split hashes and results must be archived with the revised submission, and are not claimed to be published at that locator yet.')
    old = 'Finally, the present results use the available 16-channel curve set when present; a four-curve GR/RHOB/NPHI/DTC experiment was not run, which limits direct numerical comparison with the common-core benchmark of \\citet{Carvalho2026LithologyBenchmark}. External validation on a second public well-log corpus and a common-curve subset are priorities for future work.'
    new = r'The new four-curve experiment is a bounded pilot with three overlapping closed splits and one separate open-set stress, not an expanded 18-split replication. It retains missing-value rows and our split definitions; it does not reproduce the processing or cross-validation of \citet{Carvalho2026LithologyBenchmark}. Newly fitted rows use eight LightGBM threads, while reused sixteen-curve references used \texttt{n\_jobs=-1}; runtime comparisons are not made. The WellNorm correction was audited on this limited schedule, so the historical 18-split WellNorm mean remains a legacy result. Descriptor and class-conditioned shift analyses are exploratory, and their target-label-dependent quantities cannot guide deployment. External validation on a second independent public well-log corpus, validation of any review rule and population-level inference remain outstanding.'
    if old not in text:
        raise RuntimeError('Limitations source changed')
    text = text.replace(old, new)
    text = text.replace('External-corpus validation, common-curve experiments, calibrated rejection and nested well-level inference remain the principal extensions.',
        r'The bounded common-curve pilot extends the evidence to restricted log availability and exposes a missingness-mask confound in the historical WellNorm comparison. Parent-averaged effects show that the mean normalisation gain is not a universal well-level benefit, and simple descriptors do not yet support a validated selection rule. External-corpus validation, calibrated rejection and nested well-level inference remain necessary extensions.')
    text = text.replace(r'\paragraph{Pipeline versioning.}',
        r'\paragraph{New analysis provenance.} The v2 extension has a separate configuration, source/data/split hashes, per-fit JSON and a distinct pipeline tag under \path{results/v2/}. Twenty new pilot fits and one same-thread diagnostic fit are distinguished from twelve reused sixteen-curve references; no legacy row is relabelled as newly trained. Feature-matrix equivalence tests protect the reused Global/Hybrid representations. The source-reproduction gate matches five saved Global seed-42 metrics to $10^{-10}$ before the extension. The v2 code and results are local revision artifacts and must be included in the public release before submission.' + '\n\n' + r'\paragraph{Pipeline versioning.}')
    # Focus the increment on tested questions, with no new algorithm claim.
    intro = r'The extension asks two empirical questions: does the observed normalisation benefit survive a restriction to four routinely benchmarked curves, and do readily observable well descriptors identify where that benefit reverses? A matched-row pilot separates curve availability from row selection, while a no-refit analysis averages repeated test appearances within wells and related parent wells. An implementation audit also checks whether nominal feature ablations preserve the same raw missingness information. These additions test the boundaries of the framework; they do not introduce a new classifier or a validated deployment-selection rule.'
    text = text.replace('The main contributions are:', intro + '\n\nThe main contributions are:', 1)
    text = text.replace(r'\item \textbf{Reproducible evidence.}',
        r'\item \textbf{Input availability and benefit heterogeneity.} We add a matched-row four- versus sixteen-curve pilot with consistent raw missingness indicators, and parent-averaged analyses that distinguish observable descriptors from target-label-dependent post-hoc explanations.' + '\n' + r'\item \textbf{Reproducible evidence.}', 1)
    abstract_interaction = pd.read_csv(PILOT / 'relative_depth_interaction.csv')
    abstract = r'''
Automated lithofacies classification can support well-log interpretation, but within-well depth splitting can overstate performance on newly drilled wells. We evaluate a validation framework that distinguishes inductive, post-drilling and causal-prefix target-information budgets, using a fixed LightGBM backbone on FORCE~2020. Macro-averaged F1 is paired with fixed-taxonomy worst-well and worst-decile screens that jointly reflect class coverage and prediction quality. Across 18 closed-set well-held-out repetitions, the inductive model achieves $0.339\pm0.045$ Macro-F1, compared with 0.737 under a naive depth split. Post-drilling Hybrid variants reach $0.370$--$0.379$ without uniformly improving well-level tails. A matched-row pilot restricts inputs from sixteen to four curves on three closed splits, with one open-set stress reported separately. Adding relative depth gives a mean Macro-F1 increase of {{REL4}} with four curves versus {{REL16}} with sixteen. An implementation audit identifies missingness-mask loss in historical WellNorm features; corrected controls prevent attributing the entire historical Hybrid--WellNorm gap to global log levels. Averaging repeated test appearances within wells and then parents shows positive Hybrid-norel effects for 75/107 parent wells, alongside substantial losses on others. Simple observable descriptors do not yet identify these reversals reliably. The resulting reporting framework links admissible information, log availability and heterogeneous well-level benefit, while retaining interpreter review for rare facies and uncertain wells.
'''
    abstract = abstract.replace('{{REL4}}', f'${abstract_interaction.relative_depth_delta_core4.mean():+.3f}$')
    abstract = abstract.replace('{{REL16}}', f'${abstract_interaction.relative_depth_delta_full16.mean():+.3f}$')
    a_start = text.index(r'\begin{abstract}') + len(r'\begin{abstract}')
    a_end = text.index(r'\end{abstract}', a_start)
    text = text[:a_start] + abstract + text[a_end:]
    (DST / 'main.tex').write_text(text, encoding='utf-8')

    methods = r'''\subsection{Restricted-curve pilot and implementation audit}\label{sec:v2design}

The extension locks seeds 42, 43 and 45 as an exploratory closed-set pilot and keeps open-set seed 44 separate. For each seed, all four LightGBM variants (Global, WellNorm, Hybrid-norel and Hybrid) use identical source train/validation wells and labelled test depths. The four-curve set is GR, RHOB, NPHI and DTC; the comparator uses the original sixteen channels. Missing-value rows are retained, so curve restriction does not alter the labelled cohort. Training-row counts, class encoding, class weights, source-only imputation/scaling, the 180-tree early-stopping budget and validation criterion are held fixed. Newly fitted models use eight threads. No test-set tuning or additional model search is introduced.

An audit identified a feature-budget mismatch in pipeline r3: WellNorm masks were generated from normalised values after missing values had become zero, yielding all-zero masks. The new implementation instead records $m_c=\mathbb{1}\{x_c\text{ is missing}\}$ from the original observed channel for every method. Global and Hybrid retain raw columns in the old pipeline, so their feature matrices are identical under this correction; dedicated tests check both fitting and inference. We newly fit all sixteen four-curve method--seed combinations and four corrected sixteen-curve WellNorm models. Twelve saved sixteen-curve Global/Hybrid references are reused with their original provenance. One additional seed-42 WellNorm fit retains the legacy masks at the same eight-thread setting to check whether a thread-count difference explains its discrepancy. Thus the extension adds twenty pilot fits and one diagnostic fit, rather than rerunning the full twenty-split schedule.

Reported uncertainty is the sample standard deviation across the three overlapping closed splits, not a population confidence interval. Paired curve-set differences use the same seeds; open-set seed 44 is reported separately without an unknown-class detector. The four-curve cohort retains missing observations and our well splits, so this is a within-study sensitivity analysis rather than a numerical reproduction of \citet{Carvalho2026LithologyBenchmark}.

\subsection{Exploratory well-level benefit analysis}\label{sec:v2wellmethods}

Saved predictions from all eighteen closed splits in each grouping mode provide paired per-well present-class F1 changes for Hybrid-norel minus Global. We average repeated appearances within each well and then average sibling wells within each parent. This prevents frequently sampled wells or siblings from receiving multiple points in descriptive associations, but the fitted models and source wells remain shared; parent-level summaries are therefore not independent population replicates. Present-class F1 characterises within-well predictive change, while fixed-taxonomy tails retain their separate coverage-aware role.

Observable descriptors include the full-curve missing fraction (four and sixteen channels), median $|\mathrm{CALI}-\mathrm{BS}|$, median $|\mathrm{DRHO}|$, and a four-curve location difference: the mean absolute target-well curve mean minus source-labelled median, divided by source-labelled standard deviation over available curves. These descriptors may mix geology, measurement and sampling effects. True class counts, class dominance and source--target class-prior total variation require target labels and are post-hoc quantities only. Exploratory Spearman associations and selected gain/loss examples are reported without significance tests, fitted thresholds or causal attribution. A label-dependent mean-shift identity in the Supplementary material further separates class-composition and within-class terms algebraically; it does not identify geological or instrument effects.
'''
    methods += '\n' + r'Including the earlier original-source Global seed-42 reproduction fit, this revision phase executes twenty-two fits in total: one reproduction gate, twenty pilot fits and one same-thread diagnostic. The twelve archived comparator rows are not included in that execution count.' + '\n'
    (DST / 'v2_methods.tex').write_text(methods, encoding='utf-8')

    s = pd.read_csv(PILOT / 'closed_summary.csv').set_index(['curve_set', 'method'])
    changes = pd.read_csv(PILOT / 'paired_curve_set_changes.csv')
    interaction = pd.read_csv(PILOT / 'relative_depth_interaction.csv')
    masks = pd.read_csv(PILOT / 'wellnorm_mask_control.csv')
    names = {'lgbm_global': 'Global', 'lgbm_wellnorm': 'WellNorm',
             'lgbm_hybrid_norel': 'Hybrid-norel', 'lgbm_hybrid': 'Hybrid'}
    def macro(curves, method):
        r = s.loc[(curves, method)]
        return f"${r.macro_mean:.3f}\\pm{r.macro_std:.3f}$"
    delta_text = []
    for method in names:
        r = changes[(changes.method == method) & (changes.metric == 'macro')].iloc[0]
        delta_text.append(f"{names[method]} ${r.delta_mean:+.3f}$")
    raw_delta = control['metrics']['macro']['mask_delta_same_threads']
    thread_sentence = ('The same-thread legacy-mask fit reproduces the saved legacy Macro-F1 and Worst-fixed to $10^{-10}$.'
                       if control['legacy_thread_equivalence'] else
                       'The same-thread legacy-mask fit differs from the archived reference; the same-thread mask contrast is therefore distinguished from the historical discrepancy.')
    closed_mask_changes = masks[masks.regime == 'closed_set'].delta_macro
    results = r'''\subsection{Log-availability sensitivity with consistent missingness indicators}\label{sec:v2pilot}

Table~\ref{tab:v2core} reports the bounded matched-row pilot. With four curves, closed-split Macro-F1 is {{CORE}} for Global / corrected WellNorm / Hybrid-norel / Hybrid; the corresponding sixteen-curve values are {{FULL}}. The paired four-minus-sixteen Macro changes are {{DELTAS}}. These are three-split sensitivity estimates, not a new eighteen-split headline or a ranking against a different published preprocessing pipeline. Figure~\ref{fig:v2core} also shows that pooled and coverage-aware worst-well changes need not agree.

Within the four-curve setting, the mean Hybrid--Hybrid-norel Macro difference from adding relative depth is {{REL4}}, compared with {{REL16}} under sixteen curves. Their mean difference of paired differences is {{INTERACTION}}. This input-set-dependent comparison is descriptive: the selected seeds, correlated features and logged intervals do not support a general geological mechanism or a universal benefit of relative depth. Seed~44 remains an open-set stress (Supplementary Table~\ref{tab:s-v2open}).

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Matched-row log-availability pilot, closed seeds 42/43/45. Values are mean$\pm$sample standard deviation across three overlapping splits. Raw missingness indicators are retained. Full16 Global, Hybrid-norel and Hybrid references are reused; all core4 rows and full16 WellNorm rows are newly fitted. $C_{\mathrm{common}}$ is the shared closed-training class-name set.}
\label{tab:v2core}
\centering\small
\input{table_v2_common_core}
\end{table}

\begin{figure}[width=.98\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig_v2_common_core.pdf}
\caption{Restricted-curve sensitivity under matched labelled rows. Error bars describe split standard deviations, not independent-well population uncertainty. Worst-fixed jointly reflects class coverage and prediction quality.}
\label{fig:v2core}
\end{figure}

\paragraph{Missingness-mask correction.} On seed 42, corrected sixteen-curve WellNorm Macro-F1 is {{NEW42}} versus {{OLD42}} in the legacy pipeline. {{THREAD}} At the same eight-thread setting the raw-mask change gives {{MASKDELTA}} Macro-F1. Across the three closed pilot splits, new-minus-archived WellNorm Macro changes average {{MASKMEAN}} (range {{MASKMIN}} to {{MASKMAX}}; positive in only one of three splits); the eight-thread diagnostic is restricted to seed 42. Restoring the same raw information is therefore a fairness correction, not a claim of uniform performance improvement. Full pilot results are in Supplementary Table~\ref{tab:s-v2mask}. The historical Hybrid--WellNorm gap cannot be attributed solely to retaining basin-scale log levels. The pilot supplies a corrected comparison at fixed relative-depth status (Hybrid versus WellNorm), but its limited seed count cannot replace the full historical WellNorm schedule or establish that one normalisation strategy is intrinsically superior.

\subsection{Benefit heterogeneity across wells}\label{sec:v2heterogeneity}

After averaging repeated test appearances within well and then within parent, Hybrid-norel exceeds Global in present-class F1 for 75/107 parent wells under the well-id schedule. The parent-mean paired change is $+0.027$, the median $+0.024$, and the 10th--90th percentiles $-0.043$ to $+0.100$. The parent-grouped companion has 78/106 positive parent effects, mean $+0.027$ and median $+0.017$. These summaries weight parent wells equally and thus differ from pooled split Macro-F1. They show that a positive average effect coexists with losses on individual wells (Fig.~\ref{fig:v2heterogeneity}).

Simple descriptors provide limited explanation. Under the well-id schedule, Spearman association with the paired effect is $-0.150$ for four-curve location difference, $+0.109$ for four-curve missingness, $-0.078$ for sixteen-curve missingness, $+0.034$ for median $|\mathrm{CALI}-\mathrm{BS}|$ and $+0.178$ for median $|\mathrm{DRHO}|$ (available parent counts 73--107). True class-prior total variation gives $-0.030$ and requires target labels. These exploratory associations do not validate an advance warning or model-selection rule. The label-dependent decomposition also gives small and grouping-sensitive associations: composition magnitude $-0.086$ / $-0.160$ and within-class magnitude $-0.245$ / $-0.142$ for well-id / parent-grouped schedules (Supplementary Fig.~\ref{fig:s-v2decomp}). They do not separate calibration drift from geological change causally.

\begin{figure}[width=.98\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig_v2_well_heterogeneity.pdf}
\caption{Exploratory heterogeneity of Hybrid-norel minus Global using saved sixteen-curve predictions. Points represent parent-level averages after averaging repeated appearances within each well. Panel D requires true target labels and is post-hoc only. Shared training wells and fitted models preclude interpreting points as independent population replicates.}
\label{fig:v2heterogeneity}
\end{figure}
'''
    tokens = {'CORE': ' / '.join(macro('core4', m) for m in names),
              'FULL': ' / '.join(macro('full16', m) for m in names),
              'DELTAS': ', '.join(delta_text),
              'REL4': f"${interaction.relative_depth_delta_core4.mean():+.3f}$",
              'REL16': f"${interaction.relative_depth_delta_full16.mean():+.3f}$",
              'INTERACTION': f"${interaction.difference_of_differences.mean():+.3f}$",
              'NEW42': f"${masks.loc[masks.seed == 42, 'raw_mask_macro'].iloc[0]:.3f}$",
              'OLD42': f"${masks.loc[masks.seed == 42, 'legacy_macro'].iloc[0]:.3f}$",
              'THREAD': thread_sentence, 'MASKDELTA': f'${raw_delta:+.3f}$',
              'MASKMEAN': f'${closed_mask_changes.mean():+.3f}$',
              'MASKMIN': f'${closed_mask_changes.min():+.3f}$',
              'MASKMAX': f'${closed_mask_changes.max():+.3f}$'}
    for key, value in tokens.items():
        results = results.replace('{{' + key + '}}', value)
    # SI is a separately compiled document; use its verified S-number convention.
    results = results.replace(r'\ref{tab:s-v2open}', 'S11')
    results = results.replace(r'\ref{tab:s-v2mask}', 'S12')
    results = results.replace(r'\ref{fig:s-v2decomp}', 'S10')
    (DST / 'v2_results.tex').write_text(results, encoding='utf-8')

    long = pd.read_csv(PILOT / 'matched_metrics_long.csv')
    extra = [r'\section*{Restricted-curve pilot: per-seed detail}',
             r'The four-curve and sixteen-curve rows use the same labelled depths, train/validation/test wells and training budget within each seed. Missing rows are retained. Methods: G = Global; WN = corrected WellNorm with raw masks; HN = Hybrid-norel; H = Hybrid. The sixteen-curve G/HN/H rows are reused references; all remaining rows are newly fitted.']
    short = dict(zip(names, ['G', 'WN', 'HN', 'H']))
    for regime, label in [('closed_set', 'closed'), ('open_set', 'open')]:
        rows = []
        for _, r in long[long.regime == regime].iterrows():
            rows.append([int(r.seed), r.curve_set, short[r.method], f'{r.macro:.3f}',
                         f'{r.macro_common:.3f}', f'{r.worst_fixed:.3f}', f'{r.w10_fixed:.3f}'])
        extra += [r'\begin{table}[width=.98\linewidth,pos=htbp]',
                  r'\caption{Matched-row pilot: ' + ('closed seeds 42/43/45.' if label == 'closed' else 'open-set seed 44; unknown target classes are retained and never predicted. No rejection mechanism is added.' ) + '}',
                  r'\label{tab:s-v2' + label + '}', r'\centering\small',
                  table(['Seed', 'Curves', 'Method', 'Macro', r'$C_{\mathrm{common}}$', 'Worst-fixed', 'W10-fixed'], rows, 'rllcccc'),
                  r'\end{table}']
    rows = [[int(r.seed), r.regime.replace('_', r'\_'), f'{r.legacy_macro:.3f}',
             f'{r.raw_mask_macro:.3f}', f'{r.delta_macro:+.3f}',
             f'{r.legacy_worst_fixed:.3f}', f'{r.raw_mask_worst_fixed:.3f}'] for _, r in masks.iterrows()]
    extra += [r'\begin{table}[width=.98\linewidth,pos=htbp]',
              r'\caption{Sixteen-curve WellNorm: archived legacy masks versus newly fitted raw masks. Legacy references used all available threads; new fits use eight. The seed-42 same-thread diagnostic is reported in the main text. These four pilot seeds do not replace the historical eighteen-closed-split mean.}',
              r'\label{tab:s-v2mask}', r'\centering\small',
              table(['Seed', 'Regime', 'Legacy Macro', 'Raw Macro', r'$\Delta$', 'Legacy worst', 'Raw worst'], rows, 'rlccccc'),
              r'\end{table}', r'\FloatBarrier',
              r'\section*{Exploratory post-hoc well cases and log-shift decomposition}',
              r'The six wells below are selected post-hoc as the three largest losses and gains among wells with at least two test appearances on the closed well-id schedule. They illustrate heterogeneity, not a prospectively validated rule. The effect is Hybrid-norel minus Global present-class F1 averaged over appearances.']
    cases = pd.read_csv(ANALYSIS / 'posthoc_case_wells.csv')
    rows = [[r.well_id.replace(' ', '~'), int(r.n_test_appearances), f'{r.delta_present_f1:+.3f}',
             f'{r.missing4:.3f}', f'{r.curve_location_drift4:.3f}'] for _, r in cases.iterrows()]
    extra += [r'\begin{table}[width=.95\linewidth,pos=htbp]',
              r'\caption{Post-hoc gain/loss examples. Missing4 is the full-curve missing fraction; location difference is the dimensionless four-curve source-scaled descriptor.}',
              r'\label{tab:s-v2cases}', r'\centering\small',
              table(['Well', 'Appearances', r'Mean $\Delta$ F1', 'Missing4', 'Location difference'], rows, 'lrccc'),
              r'\end{table}',
              '', r'\FloatBarrier', '',
              r'\paragraph{Class-conditioned mean-shift identity.}', '',
              r'For a curve observed in both source and target, let $q_{s,k}$ and $q_{t,k}$ denote class proportions on that curve\textquotesingle s non-missing labelled support, with class means $\mu_{s,k}$ and $\mu_{t,k}$. Then',
              r'\[\bar{x}_t-\bar{x}_s=\sum_k(q_{t,k}-q_{s,k})\mu_{s,k}+\sum_kq_{t,k}(\mu_{t,k}-\mu_{s,k}).\]',
              r'The first term is the composition term relative to source class means; the second is the within-class term. Terms are divided by the source observed standard deviation, their absolute magnitudes are averaged over available core curves, and repeated appearances are averaged within well and parent. Cases without an observed source mean for a target-observed class are omitted for that curve. The identity is checked on 2460 curve records (maximum source-standardised residual $5.91\times10^{-15}$). This is an algebraic description using true target labels. The within-class term can reflect fluids, geology, measurement, sampling and labelling; it is not an identified tool-calibration effect.',
              r'\begin{figure}[width=.98\linewidth,pos=htbp]', r'\centering',
              r'\includegraphics[width=\linewidth]{figs/fig_v2_shift_decomposition.pdf}',
              r'\caption{Post-hoc log-mean-shift components versus normalisation benefit. Parent-level averages on the well-id schedule; all components require true target labels. No thresholds, causal interpretation or significance tests are claimed.}',
              r'\label{fig:s-v2decomp}', r'\end{figure}']
    supp = (DST / 'supplementary.tex').read_text(encoding='utf-8')
    supp = supp.replace(r'\newcommand{\revmark}[1]{\textcolor{red}{#1}}', r'\newcommand{\revmark}[1]{#1}')
    supp = supp.replace('Protocol-1 method variants / hybrid ablations (seed~42).',
                        'Historical pipeline-r3 method variants (seed~42). WellNorm masks lost raw missingness information; the corresponding gaps are confounded. Corrected pilot results are reported separately.')
    supp = supp.replace('Optional seed-level intervals for the single-factor Hybrid-norel$-$WellNorm-norel Macro deltas',
                        'Archived optional seed-level intervals for the mask-confounded Hybrid-norel$-$WellNorm-norel Macro deltas')
    supp = supp.replace(r'\end{document}', '\n'.join(extra) + '\n\n' + r'\end{document}')
    (DST / 'supplementary.tex').write_text(supp, encoding='utf-8')
    (DST / 'highlights.txt').write_text(
        'Well-held-out validation separates admissible target-well information budgets.\n'
        'A matched-row pilot tests four- versus sixteen-curve log availability.\n'
        'A missingness-mask audit qualifies historical normalisation contrasts.\n'
        'Average gains coexist with losses on individual held-out wells.\n'
        'Simple well descriptors do not yet support a validated selection rule.\n', encoding='utf-8')
    report = [
        '# V2 修订说明与结果解读',
        '',
        '本轮按“现有分析＋小规模四曲线实验”完成，单独保存在 paper/gse_v2；候选 V1 和原 paper/gse 保留。',
        '',
        '实际执行训练共 22 次：原始源码复现 1 次、四曲线实验 16 次、修正缺失标记的十六曲线 WellNorm 4 次、同线程诊断 1 次。另复用 12 条十六曲线 Global/Hybrid 系参考结果，不将其计作新训练。',
        '',
        '闭集比较只有固定种子 42/43/45；44 单独作为开放集压力测试。各方法和曲线集保持相同训练、验证及评分行，保留缺失观测，不筛成完整曲线子样本。下表的标准差描述重叠划分敏感性，不是总体置信区间。',
        '',
        '| 曲线集 | 方法 | Macro-F1（均值±划分标准差） | Worst-fixed | W10-fixed |',
        '|---|---|---|---|---|',
    ]
    for curves in ['full16', 'core4']:
        for method in names:
            r = s.loc[(curves, method)]
            report.append(f'| {curves} | {names[method]} | {r.macro_mean:.3f} ± {r.macro_std:.3f} | {r.worst_fixed_mean:.3f} | {r.w10_fixed_mean:.3f} |')
    report += [
        '',
        f'四曲线条件下，Hybrid 加入相对深度的平均 Macro 收益为 {interaction.relative_depth_delta_core4.mean():+.3f}，十六曲线下为 {interaction.relative_depth_delta_full16.mean():+.3f}。这说明对相对深度的依赖随输入曲线集变化，但三次闭集划分不足以支持普适机制或推广保证。',
        '',
        '修正了一个实质性实现差异：旧 WellNorm 在缺失值已经变成零之后生成缺失标记，导致原始缺失信息消失。新实验从原始观测生成缺失标记；旧结果保留并明确标注为历史实现。旧 Hybrid－WellNorm 差距不能全部归因于保留全局测井水平。',
        '',
        f'种子 42 的同线程对照：修正缺失标记使 Macro-F1 改变 {raw_delta:+.6f}；旧实现改变线程数后是否与原结果一致：{control["legacy_thread_equivalence"]}。完整控制数据见 results/v2/mask_thread_control/control_report.json。',
        f'三个闭集种子的新旧 WellNorm 差值平均为 {closed_mask_changes.mean():+.3f}，范围 {closed_mask_changes.min():+.3f} 至 {closed_mask_changes.max():+.3f}，只有 1/3 为正。修正的目的是使缺失信息预算一致，不能宣传为普遍提高准确率。',
        '',
        '逐井分析将重复测试出现先在井内平均，再在母井内平均。well-id 方案中 75/107 个母井的 Hybrid-norel－Global 收益为正，平均 +0.027、中位数 +0.024，10%—90% 分位为 −0.043 至 +0.100。母井仍共享训练井和拟合模型，因此不能当作独立总体重复。',
        '',
        '缺失率、曲线位置差异、井径差和密度校正量等简单描述符的相关性较弱。真实类别比例及类别条件均值分解使用目标标签，只能事后解释。本轮没有建立经验证的提前预警或模型选择规则，也没有补做第二数据集。',
        '',
        '论文的实质提升是：在固定行和缺失信息预算下检验曲线可用性，披露并修正旧比较的实现混杂，并给出平均收益的井级边界。其定位仍为验证方法与实证研究；不能据此保证达到 GSE 的创新门槛或被接收。',
        '',
        '复现验证：原 Global 种子 42 五项指标与已存结果在 1e-10 内一致；20 项旧测试与 7 项新增缺失标记/特征等价/匹配行测试已通过。候选边界核验记录为 results/v2/revision_verification.json。',
        '',
        '本机脚本不调用 OpenAI API。可在独立 PowerShell 中运行 scripts/run_v2_phase.ps1；已有运行锁时会退出，避免重复启动。训练耗本机 CPU/内存；Codex 对话、读日志和分析另消耗 Codex 额度。',
    ]
    (DST / 'REVISION_V2_CN.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Built separate v2 sources: {DST}')


if __name__ == '__main__':
    main()
