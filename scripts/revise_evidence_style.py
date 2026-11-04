"""Evidence-linked English revision; preserve experimental assets and numeric tables."""
from pathlib import Path
import collections
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse_polished_20261103'
DST = ROOT / 'paper/gse_evidence_20261104'
EDITS = []
PARETO_CAPTION_OLD = r'Hybrid$-$HN Worst-fixed can flip relative to the well-id panel (Fig.~\ref{fig:pareto}).'
PARETO_CAPTION_NEW = r'Hybrid$-$HN mean Worst-fixed changes from $-0.004$ on well-id splits to $+0.006$ under parent grouping (Table~\ref{tab:pairedmr}; Fig.~\ref{fig:pareto}).'


def replace(name, old, new, reason):
    path = DST / name
    text = path.read_text(encoding='utf-8')
    assert text.count(old) == 1, (name, old[:100], text.count(old))
    path.write_text(text.replace(old, new), encoding='utf-8')
    EDITS.append({'file': name, 'original': old, 'revised': new, 'reason': reason})


def para(name, prefix, new, reason):
    text = (DST / name).read_text(encoding='utf-8')
    start = text.index(prefix)
    end = text.find('\n\n', start)
    if end < 0:
        end = len(text)
    boundary = re.search(r'\n\\(?:\[|begin\{(?:table|figure)|(?:sub)*section\*?\{|end\{abstract\})', text[start:end])
    if boundary:
        end = start + boundary.start()
    replace(name, text[start:end], new.strip(), reason)


def intro():
    para('main.tex', 'Splitting depths from the same wells', r'''
Cross-well lithofacies scores alone do not specify target-well information access or log availability. We propose a FORCE~2020 evaluation framework that separates inductive, post-drilling and causal-prefix settings and pairs Macro-F1 with coverage-aware fixed-taxonomy well tails. Across 18 closed-set well-held-out repetitions, a fixed LightGBM classifier achieves Macro-F1 $0.339\pm0.045$, compared with 0.737 under a naive depth split. Post-drilling Hybrid variants achieve $0.370$--$0.379$, while their worst-well gains vary across representations and grouping schemes. A matched-row pilot compares four and sixteen curves with consistent raw missingness indicators. On three new parent-grouped partitions of the same corpus, relative depth increases mean Macro-F1 by $+0.019$ with four curves and $+0.008$ with sixteen. The paired difference averages $+0.011$ and is positive in two of three partitions. Controlled inference tests change the relative-depth coordinate on identical retained depths while fixing all other features. These changes reduce mean four-curve Macro-F1 by $0.011$--$0.052$ across the six boundary stresses and reverse the mean advantage over Hybrid-norel under bottom and two-sided 20\% crops. We demonstrate that the measured benefit depends on both the available curves and the interval coordinate in these FORCE partitions. Reporting these conditions alongside well-held-out scores makes the comparison traceable to the information supplied to each classifier.
''', 'Rebuild the abstract as gap, framework, quantified results and significance. Replace generic future-work and self-limitation statements with measured findings and explicit experimental conditions.')
    para('main.tex', 'Wireline logs constrain', r'''
Wireline logs constrain subsurface lithology between cores and seismic volumes \citep{Asquith2004Basic,Rider2011WellLogs}. Automated lithofacies classification builds on statistical discrimination \citep{Busch1987Lithology}, neural networks \citep{Rogers1992NN,Imamverdiyev2019CNN} and tree ensembles \citep{Hall2016Facies}. FORCE~2020 provides labelled North Sea wells for reproducible comparisons \citep{Bormann2020FORCE,Merembayev2021Energies,Mousavi2024GrowNet}.
''', 'Remove an uncited assertion about manual interpretation and place citations directly after each background claim.')
    para('main.tex', 'A common evaluation draws', r'''
Depth-point splitting shares well-specific information between training and testing. Nearby samples carry correlated borehole conditions, tool responses and facies succession, which motivates grouped validation in spatial and hierarchical data \citep{Roberts2017CV}. Well-held-out lithology studies implement this separation directly \citep{Chen2025FrontiersLithology}. In our FORCE diagnostic, Global-LGBM scores 0.737 under random depth-point splitting and 0.497 under contiguous within-well depth-block hold-out (Table~\ref{tab:matched}).
''', 'Open with the protocol claim and immediately support it with methodological references and the measured diagnostic contrast.')
    para('main.tex', 'Holding out wells better', r'''
Well hold-out exposes distribution shifts between source and target wells. Calibration, washout, mud systems and facies proportions change log-response distributions \citep{Asquith2004Basic,Rider2011WellLogs,Liu2020WellDrift}, matching the source--target distinction in domain-shift theory \citep{BenDavid2010Theory,Pan2010Transfer}. FORCE training and test wells differ in both class proportions and mean curve responses (Supplementary Figs.~S1--S2).
''', 'Replace a general warning with a directly stated shift claim and corpus-specific evidence.')
    para('main.tex', 'Our focus is the evidence', '', 'Delete rhetorical questions that repeat the following framework paragraph. The protocol references remain in Related work and Discussion; the next paragraph states the contribution and numerical evidence directly.')
    para('main.tex', 'We present a reproducible FORCE', r'''
We propose an information-budget-aware reporting framework with well-held-out protocols, fixed-taxonomy tails and a machine-readable reporting record (Tables~\ref{tab:methods}, \ref{tab:checklist} and~\ref{tab:accept}). A fixed LightGBM family separates representation comparisons from per-method hyperparameter search (Section~\ref{sec:method}). Across 18 closed splits, post-drilling Hybrid reaches mean Macro-F1 0.379 versus 0.339 for inductive Global-LGBM; their fixed-taxonomy worst-well scores are 0.101 and 0.093 (Table~\ref{tab:multirep}).
''', 'State the framework contribution directly, attach its implementation and performance evidence, and remove an introductory deployment limitation.')
    para('main.tex', 'We also examine whether', r'''
We quantify how log availability and interval definitions change the measured feature benefit. A matched-row four- versus sixteen-curve pilot preserves raw missingness indicators (Table~\ref{tab:v2core}). Three new parent-grouped partitions then test the relative-depth contrast, and controlled inference stresses change the coordinate on identical retained depths (Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}). Parent-averaged effects describe benefit heterogeneity across wells (Fig.~\ref{fig:v2heterogeneity}).
''', 'Replace exploratory intent and early self-limitations with the executed design and linked evidence.')
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    start = text.index('The main contributions are:')
    end = text.index(r'\section{Related work}', start)
    replace('main.tex', text[start:end], r'''The main contributions are:
\begin{enumerate}
\item \textbf{An information-budget-aware evaluation framework.} We define admissible target-well information for inductive, post-drilling and causal-prefix prediction, and implement a machine-readable reporting record (Fig.~\ref{fig:method}; Tables~\ref{tab:methods} and~\ref{tab:accept}).
\item \textbf{Joint assessment of pooled performance and well tails.} We report Macro-F1, fixed-taxonomy worst-well F1 and W10 across 18 closed splits. Hybrid achieves mean Macro-F1 0.379 versus 0.370 for Hybrid-norel, while its worst-well score is 0.101 versus 0.105 (Table~\ref{tab:multirep}; Fig.~\ref{fig:pareto}).
\item \textbf{Matched evidence on input and interval dependence.} We compare four and sixteen curves with consistent raw masks and test relative depth on three new parent-grouped partitions. Controlled coordinate changes reverse the mean four-curve advantage over Hybrid-norel under two 20\% crops (Table~\ref{tab:v2core}; Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}).
\item \textbf{Traceable evidence across splits and wells.} We report the 18-closed/2-open partition of the locked 20-seed schedule, parent-grouped sensitivity and parent-averaged feature effects, with saved predictions and execution manifests (Tables~\ref{tab:multirep}--\ref{tab:pairedmr}; Fig.~\ref{fig:v2heterogeneity}; Data and Code availability).
\end{enumerate}

''', 'Give every contribution a concrete implementation, measurement or figure/table anchor; include the new-partition and coordinate evidence in the contribution list.')
    para('main.tex', 'Cross-well lithology under distribution', r'''
Cross-well lithology under distribution shift has established UDA and DG precedents on regional datasets \citep{Chang2021GRSL,Xu2022JNGSE,Sun2023Energies,Li2024TAI}. These studies distinguish label access, source selection and domain alignment, providing the context for our explicit information settings.
''', 'Attach the literature evidence to the paragraph topic sentence instead of postponing it to the following paragraphs.')
    para('main.tex', r'\citet{Chang2021GRSL} combine', r'''
\citet{Chang2021GRSL} combine active learning with source reweighting, increasing the training influence of source wells resembling a sparsely labelled target. \citet{Xu2022JNGSE} develop domain generalisation for interpretation while drilling, and \citet{Xu2022ICAICA} examine meta-learning under well shift. Our Hybrid-while comparison isolates a representation change: it replaces whole-well $z$-score moments with expanding prefix $\mu$/$\sigma$ and omits relative depth (Section~\ref{sec:while}; Table~\ref{tab:whilems}).
''', 'Describe prior methods with active verbs and define the present comparison through its actual feature change and evidence.')
    replace('main.tex', 'whereas UDA-lite may use unlabelled target curves', 'whereas UDA-lite permits unlabelled target curves during training', 'Replace a permission modal with the explicit information-access definition; no uncertainty is involved.')
    para('main.tex', 'The strong class imbalance', r'''
FORCE class imbalance motivates Macro-F1 alongside class-specific and well-level summaries (Fig.~\ref{fig:class}; \citep{Japkowicz2002Imbalance,Sokolova2009Metrics}). We retain interpreter-confidence scores in the parsed data and train with \texttt{class\_weight=balanced}. Section~\ref{sec:limitations} specifies the untested confidence-weighting extension.
''', 'Support the imbalance claim immediately and relocate the untested extension to the Limitations discussion.')
    para('main.tex', 'Facies generally change', r'''
We evaluate a centred majority vote of width~7 as an offline continuity heuristic (Section~\ref{sec:method}; Supplementary Fig.~S5). Geological interpretations use vertical context \citep{Avseth2010QSI,Rider2011WellLogs}; our window sensitivity measures the effect of this particular voting rule on the scored depths.
''', 'Replace an unsupported sampling-rate generalisation with the actual evaluated operation and its sensitivity evidence.')


def methods():
    replace('main.tex', r'unlabelled $\{x_i,d_i\}$ from $W_t$ may additionally enter training', r'training additionally permits unlabelled $\{x_i,d_i\}$ from $W_t$', 'State the UDA-lite permission directly rather than presenting it as scientific uncertainty.')
    para('main.tex', 'Depth samples pooled over wells', r'''
We split pooled depths uniformly at random by row, without grouping by well or blocking within-well depth. This baseline shares well identity across training and testing. We compare its score with depth-block and well-held-out scores to quantify protocol sensitivity (Table~\ref{tab:matched}; Fig.~\ref{fig:leak}).
''', 'Use an explicit actor and state what the diagnostic measures, with a direct result reference.')
    replace('main.tex', 'The authors selected these contrasts before final aggregation; companion scripts are provided.', 'We selected these contrasts before final aggregation and include their scripts in the review package.', 'Replace third-person and passive constructions with an active description of the executed procedure.')
    replace('main.tex', r'unlabelled $W_t$ can enter training', r'training permits unlabelled $W_t$', 'Express a defined information permission directly, rather than as an ability modal.')
    replace('main.tex', r'Readers should treat Table~\ref{tab:multirep}', r'The primary evidence chain comprises Table~\ref{tab:multirep}', 'State the evidence hierarchy directly instead of instructing readers how to interpret it.')
    replace('main.tex', r'and the Pareto panel (Fig.~\ref{fig:pareto}) as the primary evidence chain.', r'and the Pareto panel (Fig.~\ref{fig:pareto}).', 'Complete the declarative evidence-chain sentence after replacing its reader-directed opening.')
    para('main.tex', 'W10 averages the worst', r'''
W10 averages the worst 10\% of wells under fixed taxonomy: with $n_{\mathrm{test\ wells}}=17$, it uses $\lceil0.1\times17\rceil=2$ wells instead of a single minimum. Pooled Macro-F1 includes the never-predicted unk class whenever test truth contains it, as on seed~44. Historical $n=3$ seed-delta bootstrap intervals describe ablations on 42/43/45. Primary paired changes and win counts use $n=18$ overlapping closed repetitions and quantify split sensitivity (Table~\ref{tab:pairedmr}). We retain accuracy and weighted F1 as secondary measures because the class distribution is skewed (Fig.~\ref{fig:class}; \citep{Japkowicz2002Imbalance}); Section~\ref{sec:forcecmp} discusses the FORCE penalty matrix.
''', 'Remove the unmeasured claim that W10 is less brittle, distinguish a literal class-presence condition from the phrase "it appears", and define the actual aggregation units.')
    para('main.tex', 'Separate companions include', r'''
We include Global-RF ($\leq$120k), equal-row Global-RF (full), Global-XGB and appendix CORAL-MLP as companion configurations (Table~\ref{tab:methods}). Across the historical three closed seeds, equal-row RF-full achieves mean Macro-F1 0.402 versus 0.371 for Global-LGBM (Table~\ref{tab:multiseed}). We use LightGBM to hold the classifier family fixed in the Hybrid comparisons and evaluate centred smoothing as a separate offline step.
''', 'Replace a possibility statement with the measured comparator scores and state the fixed-backbone design directly.')
    para('main.tex', r'Let $\mathcal{C}$ denote', r'''
Hybrid combines global curve levels with within-well standardised responses. Absolute log levels contain petrophysical information and well-specific measurement offsets \citep{Asquith2004Basic,Rider2011WellLogs}. Global standardisation retains between-well levels, while well-wise $z$-scores centre and scale responses within each well. For retained curves $\mathcal{C}$, including GR, RHOB, NPHI, DTC, resistivities, CALI and PEF, we construct the following features (Fig.~\ref{fig:method}):
''', 'Open with the representation definition and remove an unisolated claim about discarding basin-scale information.')
    replace('main.tex', r'binary missingness indicators $m_c\in\{0,1\}$ after median imputation of incomplete channels.', r'binary missingness indicators $m_c\in\{0,1\}$ recorded before median imputation; the corrected pilot reads them from the original observed channels (Section~\ref{sec:v2design}).', 'Correct a prose sequencing inconsistency against src/features.py: masks precede SimpleImputer; corrected WellNorm reads the raw channel. No implementation or experiment changes.')
    para('main.tex', 'Hybrid combines between-well', r'''
Hybrid combines between-well levels ($\tilde{x}_c$) with within-well responses ($z_{c,w}$). Relative depth additionally uses the observed interval's minimum and maximum. Our controlled boundary stresses test dependence on those bounds while fixing the other features (Section~\ref{sec:v3boundarydesign}; Fig.~\ref{fig:v3boundary}). Hybrid therefore uses the post-drilling information setting; Hybrid-while omits relative depth.
''', 'Replace speculative logging scenarios with the executed coordinate test and the exact information-setting definition.')
    replace('main.tex', r'Multiclass LightGBM \citep{Ke2017LightGBM} is used as the default classifier. Training follows the same outer loop for every method variant:', r'We train multiclass LightGBM \citep{Ke2017LightGBM} with the same outer loop for each representation variant:', 'Make the training actor explicit and avoid claiming an identical algorithm loop for non-LightGBM companions.')
    para('main.tex', 'Tree depth, learning rate', r'''
We hold tree depth, learning rate, early-stopping patience and the configuration family fixed across LightGBM variants (Appendix~\ref{app:impl}). The comparison changes representations and post-processing without per-method hyperparameter search. Boosted trees and random forests provide the algorithmic context \citep{Friedman2001GBM,Chen2016XGBoost,Breiman2001RF}; the implementation uses standard scientific Python tools \citep{Pedregosa2011sklearn}.
''', 'Replace passive parameter reporting and a loosely causal assertion with the exact controlled design.')
    para('main.tex', 'Window~7 was fixed', r'''
We fixed window~7 in the release schedule and report the post-hoc seed-42 window comparison separately (Supplementary Fig.~S5). Larger windows did not increase Macro-F1 in that diagnostic. The vote operates on predicted labels, so it cannot introduce a facies absent from its input predictions.
''', 'Replace a generic potential benefit with the observed sensitivity result and the deterministic property of majority voting.')
    para('main.tex', 'We do not prescribe', r'''
We use Global-LGBM as the shared classifier for Hybrid-feature ablations. Global-RF and Global-LGBM-eq ($\leq$120k) specify their row schedules separately; equal-row Global-RF (full) and Global-XGB share the Protocol-1 training rows (Tables~\ref{tab:methods} and~\ref{tab:multiseed}). Section~\ref{sec:limitations} addresses the unmatched model capacities.
''', 'Lead with the implemented comparison and consolidate algorithm-ranking limitations in the Discussion.')


def results():
    replace('main.tex', PARETO_CAPTION_OLD, PARETO_CAPTION_NEW, 'Replace a vague possibility in the Pareto caption with the observed sign change and paired-result evidence; preserve the figure asset.')
    # Lead the primary Results subsection with its measured finding, then explain provenance.
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    start = text.index(r'\paragraph{Historical WellNorm results.}')
    end = text.index('The common-class measure covers', start)
    replace('main.tex', text[start:end], r'''\paragraph{Mean performance.} Post-drilling Hybrid improves mean Macro-F1 from $0.339\pm0.045$ for inductive Global-LGBM to $0.379\pm0.037$ across 18 closed Protocol-1 well-id splits (Table~\ref{tab:multirep}). Hybrid-norel (HN) and Hybrid-while achieve $0.370\pm0.044$ and $0.378\pm0.041$, respectively. These means and sample standard deviations describe the overlapping closed splits; open seeds~44 and~127 enter separate stress and sensitivity summaries.

The same table reports $C_{\mathrm{common}}$, Worst-fixed and W10-fixed. Worst-true denotes the legacy present-class minimum (\texttt{well\_true}). Historical seed~42 and seeds~42/43/45 supply diagnostics and ablations (Section~\ref{sec:claims}).

\paragraph{Missingness provenance.} The r3 WellNorm branches computed masks after replacing missing values with zero. We retain those scores as historical diagnostics, including the Protocol-2/3 rows, and use the raw-mask pilot for corrected WellNorm comparisons (Section~\ref{sec:v2pilot}; Supplementary Tables~S17--S19). The correction leaves Global and Hybrid features unchanged, preserving their 18-split comparison.

''', 'Start Results with the primary measured effect, put all mean/SD values next to the table, and move mask provenance after the main finding.')
    para('main.tex', 'Removing sibling', r'''
Removing sibling \texttt{34/5-1 S} from seed~44 training changes Macro-F1 from $0.326$ to $0.324$ and legacy Worst-true from $0.082$ to $0.093$ (Supplementary Table~S7). The parent-grouped schedule provides the broader grouping sensitivity (Table~\ref{tab:multirepparent}).
''', 'Replace "unlikely to dominate" with the observed removal effect and its exact scope.')
    para('main.tex', 'Hybrid-while replaces completed', r'''
Hybrid-while uses expanding causal-prefix moments and omits relative depth. On seed~42, $220733/220839$ labelled test depths have $n_{\mathrm{prefix}}\geq100$ (Supplementary Table~S1). This observed prefix support defines the wireline setting evaluated here; Section~\ref{sec:limitations} addresses sparse-prefix and MWD/LWD validation.
''', 'State the observed prefix condition precisely and consolidate deployment-validation limitations in the Discussion.')
    para('main.tex', 'Macro-F1 averages hide', r'''
Rare-class errors persist despite the higher pooled score. Under Protocol-1 seed~42, Hybrid increases F1 for Sandstone and Sandstone/Shale, while Chalk and Dolomite remain near zero (Fig.~\ref{fig:perclass}). The confusion matrix assigns Chalk mainly to Limestone, Dolomite to Shale/Sandstone, and $\approx$62--65\% of true Tuff depths to Shale (Fig.~\ref{fig:rarecm}). These observed class-specific failures define the cases requiring interpreter review.
''', 'Replace "relatively high" and an unsupported attribution of the Macro-F1 ceiling with specific class-level observations and figure evidence.')
    para('main.tex', 'For interpreter triage', r'''
We define candidate review flags for contiguous runs of $\geq$3 predicted Chalk, Dolomite or Tuff depths, and Global--Hybrid disagreements spanning $\geq$5 depths. The rare-class confusion patterns motivate these candidates (Figs.~\ref{fig:perclass}--\ref{fig:rarecm}); the study does not evaluate their sensitivity or specificity. Section~\ref{sec:limitations} specifies the validation required before using them to allocate review effort.
''', 'Distinguish proposed review candidates from validated findings; remove an unsupported lower-triage hypothesis rather than strengthening it.')
    para('main.tex', 'Historical Protocol-1 timings', r'''
The historical CPU-only fits have median runtime 211\,s (IQR 174--256\,s) across $n=18$ closed well-id repetitions and six LightGBM methods. Global-LGBM takes 181\,s (143--199), Hybrid 236\,s (179--253), and HN 239\,s (184--260), as recorded in the result CSV \texttt{seconds} field. The Windows workstations typically provide 12 cores, 16 threads and $\geq$32\,GB RAM; LightGBM~4.7 and sklearn use \texttt{n\_jobs=-1}, without GPUs.
''', 'Lead with measured runtime, identify its machine-readable evidence, and remove an unquantified assertion of comparable parent-grouped runtimes.')


def discussion():
    replace('main.tex', r'\subsection{Discussion in relation to FORCE-oriented studies}\label{sec:forcecmp}', r'\section{Discussion}\label{sec:discussion}' + '\n\n' + r'\subsection{Relation to FORCE-oriented studies}\label{sec:forcecmp}', 'Separate interpretation from Experiments and make Limitations a subsection of Discussion as requested.')
    para('main.tex', r'Table~\ref{tab:forcecmp} compares', r'''
Our framework links performance to validation and information access. With a fixed LightGBM family, the FORCE results span 0.737 under naive depth splitting and $0.339\pm0.045$ under closed well hold-out; post-drilling Hybrid variants achieve 0.370--0.379 (Tables~\ref{tab:matched} and~\ref{tab:multirep}). Algorithm comparisons \citep{Merembayev2021Energies,Mousavi2024GrowNet} and foundation-model accuracy reports \citep{Qi2025WLFM} address complementary questions. Table~\ref{tab:forcecmp} compares these emphases because their metrics and splits differ.
''', 'Put the interpretation first and attach the measured protocol and representation results before comparing literature emphases.')
    para('main.tex', 'Equal-row RF and XGBoost', r'''
The companion models reinforce the need to separate a fixed-family study from an algorithm ranking: equal-row RF-full achieves mean Macro-F1 0.402 versus 0.371 for Global-LGBM across the historical three closed seeds (Table~\ref{tab:multiseed}). Their tree capacities differ (Appendix~\ref{app:impl}). We retain unequal-budget CORAL-MLP and the historical non-r3 FORCE penalty comparison as appendix and Supplementary diagnostics, respectively.
''', 'Replace "can exceed" with the actual comparison and state the capacity condition beside the interpretation.')
    para('main.tex', 'Feature gains depend', r'''
Feature benefit depends on the tested representation and partition. Hybrid raises pooled mean Macro-F1 from 0.339 to 0.379 across the 18 closed splits (Table~\ref{tab:multirep}), while HN loses present-class F1 on individual parent-averaged wells (Fig.~\ref{fig:v2heterogeneity}). The initial four-minus-sixteen-curve difference of relative-depth gains is $+0.035$; the three new parent-grouped partitions yield a mean of $+0.011$, positive in two of three partitions (Table~\ref{tab:v2core}; Fig.~\ref{fig:v3new}). These measurements establish setting-dependent gains within the evaluated FORCE partitions.
''', 'Replace "suggested" and repeated generic caveats with measured effects, comparison scope and direct evidence anchors.')
    para('main.tex', 'Boundary stresses give', r'''
Relative depth encodes the observed interval definition as well as sample position. Changing that coordinate alone reduces mean four-curve Macro-F1 in all six crops, by $0.011$--$0.052$ (Fig.~\ref{fig:v3boundary}). For bottom and two-sided 20\% crops, the matched-support mean HN comparison changes from $+0.035$/$+0.034$ to $-0.017$/$-0.018$ (Supplementary Table~S15). Holding the other features fixed attributes these within-crop prediction changes to the coordinate supplied to the fitted learner.
''', 'Attach the interval-dependence interpretation to the controlled design and exact stress effects, without implying a geological mechanism.')
    para('main.tex', 'This sensitivity has', r'''
The coordinate result defines an operational reporting requirement: state the interval bounds used to construct relative depth. The controlled comparison fixes whole-well normalisation moments and scores identical retained depths (Section~\ref{sec:v3boundarydesign}; Supplementary Tables~S15--S16). It therefore measures coordinate sensitivity within the post-drilling representation.
''', 'State the implication directly and retain its exact feature-freezing condition instead of a stack of negative disclaimers.')
    para('main.tex', 'The missingness audit imposes', r'''
Comparable feature information is essential to interpreting the normalisation contrast. Restoring raw masks changes seed-42 sixteen-curve WellNorm Macro-F1 from 0.293 to 0.356 under the same eight-thread diagnostic (Section~\ref{sec:v2pilot}; Supplementary Table~S12). We therefore use the corrected matched-row pilot for WellNorm interpretation and retain the mask-confounded history in Supplementary Tables~S17--S19. This information-budget distinction complements the protocol emphasis of broader DG evaluations \citep{Koh2021WILDS,Zhou2023DGSurvey,Gulrajani2021DomainBed}.
''', 'Support the missingness interpretation with the same-thread diagnostic and replace defensive framing with the information-comparability requirement.')
    para('main.tex', 'The filled reporting record', r'''
The minimum reporting record connects each score to its information budget, training count and well-level summaries (Table~\ref{tab:accept}; machine-readable record in \path{results/followup/}). For the 17 test wells, W10 averages the two lowest fixed-taxonomy scores. The record reports the $n=18$ closed well-id results alongside the parent-grouped companion (Table~\ref{tab:multirepparent}). Fixed-taxonomy tails retain their coverage-aware definition from Section~\ref{sec:problem}.
''', 'Describe the concrete reporting output and remove repeated certification disclaimers already covered by metric definitions and Limitations.')
    para('main.tex', 'Global-LGBM fits settings', r'''
The information budget determines which feature construction is admissible. Global-LGBM uses source-fitted parameters; Hybrid-while adds completed-wireline causal prefixes; Hybrid-norel and Hybrid use full observed target curves (Table~\ref{tab:methods}). On this corpus, nearly all seed-42 scored depths already have $n_{\mathrm{prefix}}\geq100$ (Supplementary Table~S1), which defines the support of the reported causal-prefix result.
''', 'Replace tentative operational recommendations with the exact input-access conditions and measured prefix coverage.')
    para('main.tex', 'For post-drilling interpretation', r'''
Pooled gains and fixed-taxonomy tails answer different questions. On well-id splits, Hybrid achieves higher mean Macro-F1 than HN (0.379 versus 0.370), but lower Worst-fixed (0.101 versus 0.105); parent grouping changes the mean Hybrid$-$HN Worst-fixed difference to $+0.006$ (Table~\ref{tab:pairedmr}). The class-specific failures in Figs.~\ref{fig:perclass}--\ref{fig:rarecm} support an interpreter-assistance use case, with the validation requirements below.
''', 'Replace a conditional "can change sign" statement with the observed grouping contrast and connect the operational interpretation to class-level evidence.')
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    start = text.index(r'\subsection{Limitations}')
    end = text.index(r'\section{Conclusions}', start)
    replace('main.tex', text[start:end], r'''\subsection{Limitations and future work}\label{sec:limitations}

The 18 closed repetitions and three follow-up partitions reuse FORCE wells. Their standard deviations quantify split sensitivity, not population uncertainty (Tables~\ref{tab:multirep} and~\ref{tab:v3new}). Leaving out seed~42 or including the two open splits preserves the Global--HN--Hybrid mean ordering (Supplementary Table~S2). Population-level assessment requires nested well-level or hierarchical resampling and independent-corpus validation. We fixed the seed schedule before final provenance capture without formal preregistration. The initial pilot uses three closed splits and one open stress; its missing-row policy and well splits differ from those of \citet{Carvalho2026LithologyBenchmark}.

The protocol contrasts change more than one source of dependence. The 0.737-to-0.339 comparison combines split design with well and class composition; Table~\ref{tab:matched} reports the separate diagnostic contrasts. The initial and follow-up curve experiments also change both grouping and partition. Neither comparison identifies an isolated geological mechanism or the causal effect of parent grouping. A controlled follow-up must vary grouping and partition separately across additional wells. Protocols~2--3 remain two-quadrant and six-well fixed-draw stresses (Tables~\ref{tab:p2}--\ref{tab:p3}); extending their coverage requires complete geographic and leave-one-well-out schedules.

The boundary experiment changes relative depth while retaining full-curve $z$-scores (Section~\ref{sec:v3boundarydesign}). It does not evaluate the full workflow for incomplete logs. Such validation requires truncating the observed logs, recomputing every target-dependent feature and scoring the resulting partial-log predictions. Hybrid-while likewise evaluates prefixes of completed wireline curves, with $n_{\mathrm{prefix}}\geq100$ at nearly all seed-42 scored depths (Supplementary Table~S1). Tests on sparse prefixes and measurements available during actual MWD/LWD are required before extending that result to while-drilling deployment.

Historical WellNorm scores retain mask-confounded features; corrected evidence covers the matched pilot (Table~\ref{tab:v2core}; Supplementary Tables~S12 and~S17--S19). A corrected full-schedule rerun is required to replace the historical 18-split WellNorm mean. New fits use eight threads, whereas archived references use \texttt{n\_jobs=-1}; we do not compare their runtimes. Equal-row RF/XGBoost companions match rows rather than model capacity (Appendix~\ref{app:impl}). Strong UDA, source reweighting and confidence-weighted training require their own controlled comparisons.

The classifier has no unknown-class detector or calibrated rejection for open seeds~44 and~127 (Supplementary Table~S3 and the released split summaries). The candidate review flags have no measured sensitivity, specificity or selective-prediction guarantee (Section~\ref{sec:rare}). Validation on held-out wells must evaluate these properties before model agreement or fixed-taxonomy tails guide review allocation. Fixed tails combine class coverage with prediction quality; they do not estimate deployment-failure probabilities. The well-descriptor and class-conditioned shift analyses remain descriptive: target-label-dependent quantities cannot select a method prospectively (Fig.~\ref{fig:v2heterogeneity}; Supplementary Fig.~S10). A prospective selection rule requires development and evaluation on separate wells.

''', 'Consolidate study limitations into Discussion and pair each named evidence boundary with the validation needed to resolve it; retain negative findings and exact experimental scope.')
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    start = text.index('Cross-well lithofacies scores need')
    end = text.index(r'\appendix', start)
    replace('main.tex', text[start:end], r'''We demonstrate that validation protocol and target-well information change measured cross-well lithofacies performance on FORCE~2020. A fixed LightGBM classifier achieves Macro-F1 $0.339\pm0.045$ across 18 closed well-held-out repetitions, versus 0.737 under naive depth splitting; post-drilling Hybrid variants achieve $0.370$--$0.379$ (Tables~\ref{tab:matched} and~\ref{tab:multirep}). The joint Macro-F1 and fixed-taxonomy results distinguish pooled gains from well-tail behaviour (Fig.~\ref{fig:pareto}; Table~\ref{tab:pairedmr}).

The relative-depth benefit depends on the available curves and interval coordinate in the tested partitions. Across three new parent-grouped partitions, the four-minus-sixteen-curve difference of gains averages $+0.011$, with two of three positive (Fig.~\ref{fig:v3new}; Supplementary Table~S14). Changing relative depth alone on identical retained depths reverses the mean four-curve advantage over HN under bottom and two-sided 20\% crops (Fig.~\ref{fig:v3boundary}; Supplementary Table~S15).

We provide a reporting framework that connects each score to its information budget, curve set, interval definition and coverage-aware well tails (Tables~\ref{tab:methods}, \ref{tab:checklist} and~\ref{tab:accept}). Saved predictions, configurations and execution manifests make these comparisons traceable (Data and Code availability).

''', 'End with direct, quantified conclusions and an evidence-backed deliverable statement; consolidate future validation in the preceding Limitations subsection.')


def extensions():
    para('v2_results.tex', 'Paired four-minus-sixteen', r'''
Restricting the inputs reduces mean Macro-F1 by $0.075$ for Global, $0.042$ for corrected WellNorm, $0.080$ for HN and $0.045$ for Hybrid across the three closed pilot splits (Table~\ref{tab:v2core}). The paired pooled-score and Worst-fixed changes differ under the same labelled-row support (Fig.~\ref{fig:v2core}).
''', 'Put the quantified input effect in the topic sentence and state the three-split scope directly.')
    para('v2_results.tex', 'Adding relative depth', r'''
Relative depth increases mean Macro-F1 by $+0.043$ with four curves and $+0.008$ with sixteen across the three closed pilot splits, giving a paired difference of differences of $+0.035$ (Table~\ref{tab:v2core}; Fig.~\ref{fig:v2core}). Section~\ref{sec:v3replication} tests this contrast on new parent-grouped partitions. Supplementary Table~S11 reports open seed~44 separately.
''', 'Attach the numerical pilot result to its evidence and connect it to the follow-up instead of repeating general non-universality disclaimers.')
    para('v2_results.tex', r'\paragraph{Missingness-mask correction.}', r'''
\paragraph{Missingness-mask correction.} Restoring raw masks increases seed-42 sixteen-curve WellNorm Macro-F1 from $0.293$ to $0.356$, a change of $+0.063$ (Supplementary Table~S12). At eight threads, the legacy-mask diagnostic reproduces the archived Macro-F1 and Worst-fixed to $10^{-10}$, isolating the mask change on that seed.
''', 'Lead with the measured correction effect and specify the same-thread evidence that supports its attribution.')
    para('v2_results.tex', 'The correction is not', r'''
The mask effect varies across the three closed pilot splits. New-minus-archived Macro-F1 averages $+0.014$, ranges from $-0.013$ to $+0.063$, and is positive on one split (Supplementary Table~S12). Only seed~42 has the same-thread diagnostic. We interpret the correction as restoration of comparable raw information; Section~\ref{sec:limitations} addresses the remaining full-schedule validation.
''', 'Present the observed heterogeneity first, retain its controlled-comparison boundary, and consolidate future validation.')
    para('v2_results.tex', 'The descriptors explain', r'''
The tested descriptors have weak rank associations with the paired gain. Under well-id splitting, Spearman correlations are $-0.150$ for four-curve location difference, $+0.109$ for four-curve missingness, $-0.078$ for sixteen-curve missingness, $+0.034$ for median $|\mathrm{CALI}-\mathrm{BS}|$ and $+0.178$ for median $|\mathrm{DRHO}|$, with 73--107 parents available (Fig.~\ref{fig:v2heterogeneity}; released parent-level summaries). True class-prior total variation gives $-0.030$ and requires target labels. These descriptive associations do not evaluate a prospective selection rule.
''', 'Replace causal "explain" language with the actual rank-association quantity and attach the figure and released summaries.')
    para('v3_results.tex', 'The new parent-grouped', r'''
Relative depth increases mean Macro-F1 by $+0.019$ with four curves and $+0.008$ with sixteen on the three new parent-grouped partitions (Table~\ref{tab:v3new}). The paired difference of differences averages $+0.011$, ranges from $-0.018$ to $+0.045$, and is positive in 2/3 partitions (Supplementary Table~S14).
''', 'Open with the quantified follow-up finding and preserve the sign variation immediately beside it.')
    para('v3_results.tex', r'Table~\ref{tab:v3new} gives', r'''
The follow-up reduces the mean four-minus-sixteen-curve difference of relative-depth gains from the pilot's $+0.035$ to $+0.011$ (Fig.~\ref{fig:v3new}). The figure separates the original pilot from the new parent-grouped partitions; Section~\ref{sec:limitations} addresses the changes in grouping and the reuse of FORCE wells.
''', 'State the measured comparison directly and concentrate inference limitations in Discussion without disguising partition changes.')
    para('v3_results.tex', 'Changing the boundary coordinate', r'''
Changing the relative-depth coordinate reduces four-curve mean Macro-F1 in every prespecified crop, by $0.011$--$0.052$ (Fig.~\ref{fig:v3boundary}). Sixteen-curve mean changes range from $-0.024$ to $-0.001$. Across split--crop comparisons, retained rows range from 59.1\% to 98.9\%, and 30/36 model--crop changes are negative (Supplementary Tables~S15--S16).
''', 'Use an active effect statement and anchor the retained-support and sign counts to their complete numerical tables.')
    para('v3_results.tex', 'Each change compares', r'''
The within-crop comparison holds scored depths and class denominators fixed for changed Hybrid, baseline Hybrid and HN (Supplementary Tables~S15--S16). Changes in retained class composition therefore do not explain the difference between changed and baseline predictions on that crop. The test changes relative depth only; raw curves, masks and full-curve normalisation remain fixed (Section~\ref{sec:v3boundarydesign}). We apply the prespecified stresses without tuning an inference policy on test labels.
''', 'State the controlled comparison positively, identify exactly what is fixed, and remove a modal claim without enlarging causal scope.')
    para('v3_results.tex', 'Some coordinate changes', r'''
Bottom and two-sided 20\% coordinate crops reverse the mean four-curve advantage over HN. On identical retained depths, the bottom-crop comparison changes from $+0.035$ with the original coordinate to $-0.017$ with the stressed coordinate; the two-sided comparison changes from $+0.034$ to $-0.018$ (Supplementary Table~S15). These two controlled comparisons demonstrate interval-coordinate dependence for the fitted learner.
''', 'Replace an indefinite quantifier and stacked negative qualifications with the two measured reversal cases and the fitted-model scope.')
    para('v2_methods.tex', 'Standard deviations describe', r'''
We pair curve-set differences by seed and report sample standard deviations across the three overlapping closed splits. We score open seed~44 separately (Supplementary Table~S11). Section~\ref{sec:limitations} distinguishes this design from the preprocessing and CV of \citet{Carvalho2026LithologyBenchmark} and specifies the population-inference boundary.
''', 'Make the analysis actor explicit and keep the aggregation condition while consolidating broader limitations.')
    para('v2_methods.tex', 'Observable descriptors are', r'''
We compute full-curve missing fractions for four and sixteen channels, median $|\mathrm{CALI}-\mathrm{BS}|$, median $|\mathrm{DRHO}|$, and a four-curve location difference. The last averages absolute differences between target-well curve means and source-labelled medians, each divided by its source-labelled standard deviation. These are observable log descriptors, not labels of geological or measurement causes.
''', 'Describe the computation actively and replace a generic causal possibility with the descriptor definition.')
    para('v3_methods.tex', 'Before fitting the follow-up', r'''
We fixed seeds 20261103, 20261104 and 20261105 before fitting the follow-up models and retain all three partitions. Each split keeps wellbores sharing a parent in the same training, validation or test subset. Source/test class coverage determines closed/open membership, and all three partitions are closed (Table~\ref{tab:v3new}). Section~\ref{sec:limitations} addresses the overlap of FORCE wells across partitions.
''', 'Use an explicit actor for seed selection and retain the rule for closed-set membership with its evidence.')
    para('v3_methods.tex', 'Hybrid and Hybrid-norel are fitted', r'''
We fit Hybrid and Hybrid-norel with both curve sets in each partition, giving twelve fits. We retain the earlier full-row budget, raw masks, hyperparameters and source-well validation. The primary descriptive contrast is the paired difference of differences
''', 'Replace agentless passive fitting with an active statement of the twelve-fit design while preserving the following equation.')
    para('v3_methods.tex', 'Here $H$ denotes', r'''
Here $H$ denotes Hybrid and $HN$ Hybrid-norel. Terms within a split use identical scored depths. We report each $I_s$, its mean and split standard deviation (Supplementary Table~S14). Before running the new schedule, we reproduced five v2 four-curve Hybrid seed-42 metrics to $10^{-10}$ and saved boosters, transformation state and row-level predictions.
''', 'Attach the analysis output to its numerical table and state the reproduction and saving actions explicitly.')


def appendix_and_supplement():
    replace('main.tex', 'The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.', 'The authors declare no known competing financial interests or personal relationships with an actual or perceived influence on the work reported in this paper.', 'Express the existing declaration through actual or perceived influence, retaining the no-known-interest scope without the conditional verb could.')
    para('main.tex', 'Reported in main Tables', r'''
We report Global-RF under the $\leq$120k-row schedule as an inductive reference (Tables~\ref{tab:p1} and~\ref{tab:multiseed}). Appendix~\ref{app:rf} gives its configuration and Protocol-2/3 companions. Table~\ref{tab:v2core} reports the separate raw-mask WellNorm pilot.
''', 'Replace a subjectless fragment and repeated non-superiority disclaimers with the actual reference schedule and evidence locations.')
    para('main.tex', 'Global-RF (full) trains', r'''
Global-RF (full) trains on all Protocol-1 training rows with $n_{\mathrm{estimators}}=100$, $\texttt{max\_depth}=20$ and $\texttt{min\_samples\_leaf}=50$. Global-XGB uses the same inductive global features with $n_{\mathrm{estimators}}=180$, $\texttt{learning\_rate}=0.05$, $\texttt{max\_depth}=8$ and well-held-out early stopping. Across closed seeds~42/43/45, RF-full scores 0.402/0.098 for Macro-F1/Worst-fixed, XGB 0.377/0.101 and Global-LGBM 0.371/0.104 (Table~\ref{tab:multiseed}). Equal-row comparisons match training counts, not tree depth or estimator count; the $\leq$120k RF schedule uses $n_{\mathrm{estimators}}=200$ and $\texttt{max\_depth}=24$ (Appendix~\ref{app:rf}).
''', 'Retain all configuration and score facts while deleting the unsupported expectation about inductive bias and informal defensive language.')
    replace('main.tex', 'Raw competition files can be obtained from the cited dataset source. No independent external-corpus result is claimed.', 'The cited dataset source provides the raw competition files. Section~\ref{sec:limitations} specifies the independent-corpus validation requirement.', 'Replace a modal availability statement with an active source statement and consolidate the external-validation limitation.')
    para('supplementary.tex', 'Archived optional seed-level intervals', r'''
\path{scripts/revision_r3_openset_bootstrap.py} regenerates archived seed-level intervals for the mask-confounded Hybrid-norel$-$WellNorm-norel Macro changes on seeds~42/43/45 ($B=5000$ bootstrap over three seed deltas). These intervals describe the historical three-seed diagnostic. The primary aggregation uses the locked 18-closed/2-open Protocol-1 schedule; the main-text closed multi-rep table supplies the headline means. Protocols~2--3 remain fixed-draw stresses.
''', 'Use the regeneration script as the explicit actor and distinguish the historical diagnostic from the primary aggregation concisely.')
    para('supplementary.tex', 'The identity holds across', r'''
The identity holds across 2460 curve records, with maximum source-standardised residual $5.91\times10^{-15}$. It decomposes the observed shift algebraically using true target labels. It does not identify the contributions of fluids, geology, measurement, sampling or labelling to the within-class term.
''', 'Retain the exact algebraic check and replace a speculative list of possible causes with a precise identifiability statement.')


def words(text):
    text = re.sub(r'\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}', '', text, flags=re.S)
    text = re.sub(r'\\(?:cite\w*|ref|label|path|url)\{[^}]*\}', '', text)
    text = re.sub(r'\$[^$]*\$|(?<!\\)\\\[.*?(?<!\\)\\\]', '', text, flags=re.S)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text))


def audits():
    before_all = after_all = ''
    checks = {}
    for oldpath in SRC.glob('*.tex'):
        name = oldpath.name
        old, new = oldpath.read_text(encoding='utf-8'), (DST / name).read_text(encoding='utf-8')
        def blocks(kind, text):
            return re.findall(r'\\begin\{' + kind + r'\}.*?\\end\{' + kind + r'\}', text, re.S)
        def citekeys(text):
            return {key.strip() for args in re.findall(r'\\cite\w*\{([^}]+)\}', text) for key in args.split(',')}
        checks[name] = {
            'tables_byte_identical': blocks('table', old) == blocks('table', new),
            'figures_identical_except_documented_pareto_caption': blocks('figure', old) == blocks('figure', new.replace(PARETO_CAPTION_NEW, PARETO_CAPTION_OLD)),
            'display_equations_identical': re.findall(r'(?<!\\)\\\[.*?(?<!\\)\\\]', old, re.S) == re.findall(r'(?<!\\)\\\[.*?(?<!\\)\\\]', new, re.S),
            'citation_keys_identical': citekeys(old) == citekeys(new),
        }
        assert all(checks[name].values()), (name, checks[name])
        if name in ['main.tex', 'v2_methods.tex', 'v2_results.tex', 'v3_methods.tex', 'v3_results.tex']:
            before_all += old + '\n'
            after_all += new + '\n'
    protected = [p for p in (SRC / 'figs').rglob('*') if p.is_file()]
    protected += [SRC / n for n in ['refs.bib', 'supplementary-refs.bib', 'table_v2_common_core.tex', 'table_v3_new_splits.tex', 'highlights.txt']]
    asset_checks = {p.relative_to(SRC).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256((DST / p.relative_to(SRC)).read_bytes()).hexdigest() for p in protected}
    assert all(asset_checks.values())
    numbers = r'(?<![A-Za-z])\d+(?:\.\d+)?'
    new_numbers = sorted(set(re.findall(numbers, after_all)) - set(re.findall(numbers, before_all)))
    assert not new_numbers, ('Unreviewed new numerical literals', new_numbers)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', after_all, re.S).group(1)
    banned = r'\b(?:may|might|could|possibly|potentially|somewhat|relatively|fairly|quite|arguably|unfortunately|admittedly|significantly)\b|to some extent|it seems|it appears|we believe|we think|we hope|we try to|we attempt to|it is believed that|it was shown that|due to time limits|our method is simple'
    hedge_before, hedge_after = [], []
    for base, output in [(SRC, hedge_before), (DST, hedge_after)]:
        for path in base.glob('*.tex'):
            for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                for match in re.finditer(banned, line, re.I):
                    output.append({'file': path.name, 'line': line_no, 'term': match.group(), 'context': line})
    scientific_after = [h for h in hedge_after if 'competing financial interests' not in h['context']]
    assert not scientific_after, scientific_after
    assert not re.search(banned, abstract, re.I)
    assert not re.search(r'\b(?:limitations?|future work|outstanding|further validation)\b', abstract, re.I)
    classifications = []
    for item in hedge_before:
        context = item['context']
        if 'competing financial interests' in context:
            category, action = 'Non-scientific conditional declaration', 'Retain the no-known-interest scope and express the condition as actual or perceived influence, without a modal verb.'
        elif item['term'].lower() == 'may':
            category, action = 'Information-access permission, not scientific uncertainty', 'Replace the permission modal with permits/permits during training.'
        elif item['term'].lower() == 'it appears':
            category, action = 'Literal class occurrence, not hedging', 'Rewrite as test truth contains it to remove lexical ambiguity.'
        else:
            category, action = 'Vague evaluation rather than a measured quantity', 'Replace with class-level findings and immediate figure evidence.'
        classifications.append({**item, 'classification': category, 'action': action})
    modal_review = []
    for path in DST.glob('*.tex'):
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            for match in re.finditer(r'\b(?:can|would|should|suggest(?:s|ed)?|unlikely|possible|potential|provisional)\b', line, re.I):
                if 'full-curve deployment filter' in line:
                    category = 'Conditional description of a distinct, unimplemented full-grid procedure; not a reported experimental result.'
                elif 'Authors should state' in line:
                    category = 'Reporting-checklist instruction, not uncertainty in a scientific claim.'
                elif 'Low fixed scores can coincide' in line:
                    category = 'Conditional metric property with explicit k<<K condition and a seed-139 well example in the same caption.'
                else:
                    raise AssertionError(('Unclassified remaining modal', path.name, line_no, match.group(), line))
                modal_review.append({'file': path.name, 'line': line_no, 'term': match.group(), 'context': line, 'classification': category})
    report = {
        'edited_units': len(EDITS), 'checks': checks, 'protected_asset_checks': asset_checks,
        'prose_tokens_before': words(before_all), 'prose_tokens_after': words(after_all),
        'word_count_scope': 'Same approximate English-token metric as the prior language audit; excludes table/figure blocks, math and citation/cross-reference fields but retains some preamble and labels.',
        'introduced_numerical_literals': new_numbers, 'new_experiments': 0,
        'hedges_before': classifications, 'hedges_after': hedge_after,
        'remaining_extended_modal_review': modal_review,
        'scientific_banned_hedges_after': scientific_after,
        'abstract_word_tokens': words(abstract), 'abstract_generic_limitations_or_future_work': False,
        'discussion_separate_from_experiments': r'\section{Discussion}' in after_all,
        'figure_caption_edits': 1,
        'limitation_resolution': 'Each consolidated limitation names a specific required validation; experimental conditions remain beside the corresponding claims.',
    }
    (DST / 'EVIDENCE_STYLE_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (DST / 'REWRITE_EDITS.json').write_text(json.dumps(EDITS, indent=2, ensure_ascii=False), encoding='utf-8')
    # All actual changed units are available, exceeding the requested ten examples.
    def clean(text):
        text = re.sub(r'\\(?:cite\w*|ref|label)\{([^}]+)\}', lambda m: '[' + m[1] + ']', text)
        text = text.replace('|', r'\|').replace('\n', '<br>')
        return text
    rows = ['# Rewrite comparison / 改写对照表', '',
            'Each row records an actual source edit. English wording preserves the existing results and evidence boundaries. Figure and table label keys are shown in brackets for traceability.', '',
            '| No. | Original | Revised | Reason |', '|---|---|---|---|']
    for index, edit in enumerate(EDITS, 1):
        rows.append(f'| {index} | {clean(edit["original"])} | {clean(edit["revised"])} | {edit["reason"]} |')
    (DST / 'REWRITE_COMPARISON.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['edited_units', 'prose_tokens_before', 'prose_tokens_after', 'abstract_word_tokens', 'introduced_numerical_literals', 'scientific_banned_hedges_after']}, indent=2))


def main():
    DST.mkdir(exist_ok=True)
    for path in SRC.iterdir():
        if path.suffix in {'.tex', '.bib', '.cls', '.sty', '.bst'} or path.name == 'highlights.txt':
            shutil.copy2(path, DST / path.name)
    shutil.copytree(SRC / 'figs', DST / 'figs', dirs_exist_ok=True)
    intro()
    methods()
    results()
    discussion()
    extensions()
    appendix_and_supplement()
    audits()


if __name__ == '__main__':
    main()
