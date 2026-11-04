"""Language-only revision. Keep numerical tables, equations and earlier files intact."""
from pathlib import Path
import shutil
import re
import json
import hashlib
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse_final_20261103'
DST = ROOT / 'paper/gse_polished_20261103'
EDITS = []


def paragraph(name, prefix, replacement):
    path = DST / name
    text = path.read_text(encoding='utf-8')
    start = text.index(prefix)
    end = text.find('\n\n', start)
    if end < 0:
        end = len(text)
    structural = re.search(r'\n\\(?:\[|begin\{(?:table|figure)|(?:sub)*section\*?\{|end\{document\})', text[start:end])
    if structural:
        end = start + structural.start()
    if text.rfind(r'\begin{abstract}', 0, start) > text.rfind(r'\end{abstract}', 0, start):
        end = text.index(r'\end{abstract}', start)
    old = text[start:end]
    text = text[:start] + replacement.strip() + '\n' + text[end:]
    path.write_text(text, encoding='utf-8')
    EDITS.append({'file': name, 'old': old, 'new': replacement.strip()})


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for p in SRC.iterdir():
        if p.suffix in {'.tex', '.bib', '.cls', '.sty', '.bst'} or p.name == 'highlights.txt':
            shutil.copy2(p, DST / p.name)
    shutil.copytree(SRC / 'figs', DST / 'figs', dirs_exist_ok=True)
    edit_main()
    edit_methods()
    edit_results()
    edit_ending()
    edit_supplement()
    align_titles_and_layout()
    audit()


def edit_main():
    paragraph('main.tex', 'Within-well depth splitting can overstate', r'''
Splitting depths from the same wells can give an optimistic estimate of lithofacies-classification performance on a new well. Using FORCE~2020 and a fixed LightGBM family, we examine how validation protocol, target-well information and log availability affect that estimate. We distinguish inductive, post-drilling and causal-prefix settings, and report Macro-F1 alongside coverage-aware fixed-taxonomy well tails. Across 18 closed-set well-held-out repetitions, inductive Macro-F1 is $0.339\pm0.045$, compared with 0.737 under a naive depth split. Post-drilling Hybrid variants reach $0.370$--$0.379$, but improvements in well tails are inconsistent. A matched-row pilot compares four and sixteen curves and corrects missingness-mask loss in historical WellNorm comparisons. On three new parent-grouped partitions of the same corpus, adding relative depth changes mean Macro-F1 by $+0.019$ with four curves and $+0.008$ with sixteen. Their mean paired difference is $+0.011$, with a direction that varies across partitions. Inference stresses then change interval boundaries while holding other features fixed and comparing identical retained depths. These tests reveal sensitivity to the relative-depth coordinate, rather than performance under physical incomplete logging. Parent-averaged gains also vary, and simple observable descriptors do not reliably identify reversals. The results support reporting information access, available logs and interval definitions together with cross-well scores. Independent-corpus validation and validated model-selection rules remain outstanding.''')
    paragraph('main.tex', 'Wireline logs remain a primary constraint', r'''
Wireline logs constrain subsurface lithology between cores and seismic volumes \citep{Asquith2004Basic,Rider2011WellLogs}. Interpreting them by hand is labour-intensive, and facies picks can differ among interpreters. Automated classification has a long history, from statistical discrimination and early neural networks \citep{Busch1987Lithology,Rogers1992NN} to tree ensembles and convolutional models \citep{Hall2016Facies,Imamverdiyev2019CNN}. Public datasets, particularly FORCE~2020 \citep{Bormann2020FORCE}, now make it easier to compare these methods on labelled North Sea wells \citep{Merembayev2021Energies,Mousavi2024GrowNet}.''')
    paragraph('main.tex', 'A recurring evaluation choice is to draw', r'''
A common evaluation draws training and test depths from the same wells. Nearby samples share borehole conditions, tool responses and often the same facies succession. A random row split therefore places correlated depths on both sides of the comparison, sharing information unavailable on a newly drilled well. Structured cross-validation addresses this problem in spatial and hierarchical data \citep{Roberts2017CV}; well-log studies likewise recommend holding out entire wells \citep{Chen2025FrontiersLithology}.

Holding out wells better approximates the intended use, but it exposes another difficulty. Calibration, washout, mud systems and facies proportions differ among wells, shifting the distributions of log responses \citep{Asquith2004Basic,Rider2011WellLogs,Liu2020WellDrift}. In statistical terms, the source and target wells undergo domain shift \citep{BenDavid2010Theory,Pan2010Transfer}. A strong within-well score need not survive that shift.''')
    paragraph('main.tex', 'In machine learning terms the problem is', r'''
Two machine-learning settings distinguish how target information is used. Domain generalization (DG) leaves the target well unseen during training; unsupervised domain adaptation (UDA) allows unlabelled target curves for alignment \citep{Zhou2023DGSurvey,Wang2023DGSurvey,Wilson2020UDASurvey}. Cross-well studies already investigate source reweighting, adversarial alignment and interpretation while drilling on regional datasets \citep{Chang2021GRSL,Xu2022JNGSE,Sun2023Energies,Li2024TAI} (Section~\ref{sec:related-crosswell}). For public FORCE~2020 and Geolink data, \citet{Carvalho2026LithologyBenchmark} provide a standardised benchmark using four core logs, five-fold CV and ten model families.

Our focus is the evidence needed to interpret a cross-well score. For a fixed model family, how does depth-point splitting compare with well hold-out, which target-well information is available, and what happens in the worst-performing wells? These questions complement method benchmarking. Work outside geoscience also shows how strongly protocol design can affect apparent algorithm performance \citep{Koh2021WILDS,Gulrajani2021DomainBed}.''')
    paragraph('main.tex', 'We release an information-budget-aware validation', r'''
We present a reproducible FORCE~2020 framework linking admissible target-well information to well-held-out evaluation, fixed-taxonomy tails and a machine-readable minimum reporting record (Tables~\ref{tab:checklist}--\ref{tab:accept}). LightGBM remains fixed so that the comparison centres on validation and information access rather than algorithm search (Section~\ref{sec:claims}). The framework clarifies what a score supports operationally. Macro-F1 near 0.34--0.38 and persistent rare-class failures still leave an important role for interpreter review (Section~\ref{sec:limitations}).''')
    paragraph('main.tex', 'The extension asks two empirical questions:', r'''
We also examine whether the normalisation benefit persists with only four routinely benchmarked curves, and whether observable well descriptors help identify reversals. The curve-set comparison keeps labelled rows fixed. A separate analysis averages repeated test appearances within wells and parent wells, without refitting models. An implementation audit checks whether the feature comparisons preserve the same missingness information. These analyses test where the framework applies; they supply neither a new classifier nor a validated rule for selecting one at deployment.''')
    paragraph('main.tex', 'Early work compared Bayesian discrimination', r'''
Early facies-classification studies compared Bayesian discrimination, fuzzy methods, $k$-nearest neighbours and neural networks on core-calibrated data \citep{Dubois2007Facies}. These approaches followed statistical lithology estimation \citep{Busch1987Lithology} and back-propagation networks for multi-lithology recognition \citep{Rogers1992NN}. Support vector machines also proved competitive in heterogeneous reservoirs \citep{AlAnazi2010Lithology,AlAnazi2010Lithofacies}.

Tree ensembles remain widely used tabular learners \citep{Breiman2001RF,Friedman2001GBM,Chen2016XGBoost,Ke2017LightGBM,Pedregosa2011sklearn}. Well-log comparisons find them strong baselines when channels are missing and classes are imbalanced \citep{Merembayev2021Energies}. Deep approaches include one-dimensional convolutional models for facies succession \citep{Imamverdiyev2019CNN} and convolutional models for lithology in core CT imagery \citep{Chawshin2021CT}.''')
    paragraph('main.tex', 'On FORCE~2020 and related North Sea wells,', r'''
Studies on FORCE~2020 and related North Sea data compare conventional and boosted learners using competition-style metrics \citep{Merembayev2021Energies}, or combine neural boosting with image-transformed tabular inputs \citep{Mousavi2024GrowNet}. Other work addresses transfer under log drift \citep{Liu2020WellDrift} and geology-aware prediction with whole-well hold-out \citep{Chen2025FrontiersLithology}. The FORCE penalty score differs from Macro-F1, so leaderboard ranks cannot be compared directly with our results (Section~\ref{sec:forcecmp}). We retain LightGBM as a transparent backbone and hold out wells, following the wider case for group-based evaluation of spatially dependent observations \citep{Roberts2017CV}.''')
    paragraph('main.tex', 'Cross-well lithology identification under distribution', r'''
Cross-well lithology under distribution shift is an established research area. Much of this work uses proprietary Bohai, Jiyang or other regional wells rather than FORCE~2020. The following studies are especially relevant to our information settings.''')
    paragraph('main.tex', r'\citet{Chang2021GRSL} combine active learning', r'''
\citet{Chang2021GRSL} combine active learning with source reweighting, giving more training influence to labelled source wells resembling a sparsely labelled target. Their emphasis is active querying when target labels are scarce. \citet{Xu2022JNGSE} develop domain generalisation for interpretation while drilling, while \citet{Xu2022ICAICA} examine meta-learning under well shift. Hybrid-while serves a narrower purpose here. It replaces whole-well $z$-score moments with expanding prefix $\mu$/$\sigma$ and omits relative depth, without introducing an IWD architecture or meta-learner. Source reweighting and confidence-weighted UDA-lite remain possible extensions; the present contribution organises the FORCE evidence by admissible information.''')
    paragraph('main.tex', 'Those studies establish that evaluating on unseen', r'''
These studies make clear that testing on unseen wells is already standard practice. The shared FORCE and Geolink processing benchmark of \citet{Carvalho2026LithologyBenchmark} is particularly close to our work. We complement that comparison with a fixed-family contrast between depth-point and well-held-out evaluation, explicit information budgets, and coverage-aware well tails (Table~\ref{tab:forcecmp}). Foundation-model accuracy and few-shot results \citep{Qi2025WLFM} address a related goal, but do not measure the same quantities as Macro-F1 and well-level tails.''')
    paragraph('main.tex', 'FORCE lithofacies labels are strongly imbalanced,', r'''
The strong class imbalance in FORCE makes accuracy alone difficult to interpret, motivating Macro-F1 and related summaries \citep{Japkowicz2002Imbalance,Sokolova2009Metrics}. We retain interpreter-confidence scores in the parsed data but do not use them as sample weights. The reported models use \texttt{class\_weight=balanced}; confidence-weighted training remains an untested extension.

Facies generally change more slowly along depth than the logging sample rate. We therefore examine a centred majority vote of width~7 as a continuity heuristic, with window sensitivity reported in the Supplementary material. Geological context matters when interpreting log responses \citep{Avseth2010QSI,Rider2011WellLogs}, but those principles do not prescribe this particular filter or window.''')
    paragraph('main.tex', 'The protocols, information-budget taxonomy,', r'''
The main contribution is the FORCE~2020 reporting framework: well-held-out protocols, explicit information budgets, Macro-F1, fixed-taxonomy well tails and a minimum reporting record (Tables~\ref{tab:checklist}--\ref{tab:accept}). LightGBM fixes the model family rather than establishing algorithm superiority. Figure~\ref{fig:leak} contrasts optimistic depth-point scores with well hold-out. The diagnostics in Table~\ref{tab:matched} examine separate protocol changes (A$\to$B $-0.240$; D$-C^\star$ $\approx+0.02$--$0.03$), not additive causal components of that gap.''')
    paragraph('main.tex', 'Hybrid-family effects are evaluated jointly', r'''
Feature comparisons also depend on which missingness information is retained. Pipeline r3 discarded the raw masks in its WellNorm branches, so historical Hybrid--WellNorm gaps do not isolate global curve levels. The historical $2\times2$ table is retained for traceability. The matched-row pilot in Section~\ref{sec:v2pilot} preserves raw masks for every new fit. Comparisons against inductive Global additionally differ in access to target-well information.''')
    paragraph('main.tex', 'Fig.~\\ref{fig:method} maps the released baselines', r'''
Figure~\ref{fig:method} places the baselines within their information settings. Global-LGBM uses source-fitted features; Hybrid-norel and Hybrid add whole-well normalisation, while Hybrid-while uses causal prefixes. Target lithofacies labels remain hidden until scoring. The Hybrid variants share a CPU LightGBM backbone, allowing feature comparisons without an algorithm search.

Separate companions include Global-RF ($\leq$120k), equal-row Global-RF (full) and Global-XGB, and CORAL-MLP (Table~\ref{tab:methods}). Equal-row RF-full can exceed Global-LGBM in mean Macro-F1 (Table~\ref{tab:multiseed}); LightGBM is therefore a demonstration backbone, not the winner of an algorithm ranking. Centred smoothing is an optional offline step. Broader UDA-lite comparisons remain outside the appendix CORAL schedule.''')
    paragraph('main.tex', 'Let $\\mathcal{C}$ be the set of core curves', r'''
Let $\mathcal{C}$ denote the retained core curves, including GR, RHOB, NPHI, DTC, resistivities, CALI and PEF. Absolute levels carry petrophysical information, but tool and borehole conditions can introduce well-specific offsets \citep{Asquith2004Basic,Rider2011WellLogs}. Global standardisation retains between-well levels along with these offsets. Well-wise $z$-scores remove offsets but can also discard discriminative basin-scale information. Hybrid therefore combines the following features at each depth:''')
    paragraph('main.tex', 'Global standardisation and median imputation are fit', r'''
Global scaling and median imputation are fitted on training wells and then frozen. By contrast, whole-well $z$-scores and relative depth use the scored well's full observable curve, regardless of label availability. They belong to test-domain normalisation, not inductive DG. Feature rows are emitted only at labelled training or scoring depths.

Hybrid combines between-well levels ($\tilde{x}_c$) with within-well responses ($z_{c,w}$). Relative depth further requires the observed interval's minimum and maximum depth. Incomplete runs, casing seats and differences in logged interval can distort $d^{\mathrm{rel}}$. Its intended setting is post-drilling interpretation, not causal-prefix or early-interval scoring without known TD.

The historical $2\times2$ comparison of \{well-$z$, well-$z$+$d^{\mathrm{rel}}$, global+well-$z$, global+well-$z$+$d^{\mathrm{rel}}$\} corresponds to WellNorm-norel, WellNorm, Hybrid-norel and Hybrid. Its WellNorm branches lost raw missingness indicators, preventing clean attribution to global levels. The matched pilot corrects that discrepancy (Section~\ref{sec:v2design}). Hybrid--Hybrid-norel still compares the addition of relative depth under a shared training schedule. Hybrid-while instead uses causal prefix $\mu$/$\sigma$ and omits $d^{\mathrm{rel}}$ (Section~\ref{sec:while}). Validation and test labels never enter feature fitting. These dense tabular features are suited to gradient-boosted trees \citep{Ke2017LightGBM,Chen2016XGBoost}.''')
    paragraph('main.tex', 'Centred majority voting (default window~7)', r'''
Centred majority voting (default window~7) acts only on predictions at contiguous labelled scoring depths; it never reads target label values. Gaps larger than $1.5\times$ the within-well median positive $\Delta d$ break the vote. Sparse label coverage makes this a score-support diagnostic. A full-curve deployment filter would instead predict and smooth the entire wireline MD grid before restricting the score to labelled depths.

Window~7 was fixed in the release schedule. The 5--7 window comparison on Protocol-1 seed~42 is post-hoc and does not select a production default. Smoothing can reduce isolated errors within coherent intervals, but cannot recover facies the classifier never proposes. Larger windows did not improve the Supplementary sensitivity result.''')


def edit_methods():
    paragraph('main.tex', 'Wellbores are partitioned into train, validation', r'''
Protocol-1 splits wellbores into approximately 70\% training, 15\% validation and 15\% testing, using a fixed random seed. Assignment is by wellbore count, not parent-well family, and scoring uses test wellbores only. Under our suffix-stripped parent key, just one parent has multiple wellbores in this labelled corpus (\texttt{34/5-1 A}/\texttt{34/5-1 S}). Seed~44 places these across training and testing (Section~\ref{sec:parent}). The prespecified parent-grouped companion assigns all wellbores of a parent together, using the same locked multi-rep seed list.''')
    paragraph('main.tex', 'All wells belonging to selected', r'''
Protocol-2 holds out selected NPD quadrants. Official Norwegian well names follow \texttt{QUADRANT/BLOCK-WELL}; for example, 15/9-23 lies in quadrant~15, block~9. We group by the leading quadrant token and report quadrants~15 and~35 separately. Thus, this is a coarse geographic stress, not a full quadrant/block hold-out or formal province map. Released CSVs retain the historical field \texttt{block} and tag \texttt{protocol2\_block\allowbreak\_holdout}, both referring to quadrant ids.

Quadrant~15 contains 5~labelled wells ($\approx$8.5$\times10^4$ samples; 10 facies codes), while quadrant~35 contains 21 ($\approx$2.4$\times10^5$ samples; 8 codes). Shale dominates both. Results from two quadrants and one seed are illustrative stresses (Section~\ref{sec:p2}).''')
    paragraph('main.tex', 'For each of six target wells,', r'''
Protocol-3 trains on all remaining labelled wells and tests on one target well at a time. Six targets were selected automatically by sorting wells by quadrant and sample count, then taking index-spaced entries (\texttt{max\_targets}=6). The resulting set is \{15/9-23, 17/11-1, 29/3-1, 33/9-1, 34/10-16~R, 7/1-1\}. We call this a six-well sampled stress test, rather than full leave-one-well-out over all 118 wells. Legacy filenames retain \texttt{protocol3\_lowo}. Its mean describes this fixed draw, not a population estimate (Section~\ref{sec:lowo}).''')
    paragraph('main.tex', 'LAS files from FORCE~2020', r'''
We converted the FORCE~2020 LAS files \citep{Bormann2020FORCE} into parquet tables containing well identity, depth, curves, lithofacies and confidence. The full-curve table contains about $2.34\times10^6$ depths from 118 wells; about $1.43\times10^6$ depths carry labels. Label coverage is $\approx$61\% overall. Across wells it ranges from $\approx$9\% to $\approx$100\%, with a mean of $\approx$62\% (Table~\ref{tab:coverage}). Whole-well $z$-scores, prefix moments and relative depth use all observable depths. Supervised training, early stopping and scoring use labelled depths only.

The full feature set contains 16 channels where present: GR, RHOB, NPHI, DTC, DTS, PEF, CALI, RDEP, RMED, RSHA, RXO, SP, ROP, DRHO, BS and MUDWEIGHT. Missing values are imputed with training-set medians and accompanied by masks. Smoothing respects gaps: a gap exceeding $1.5\times$ the median positive within-well $\Delta d$ breaks the vote.

We retain test facies absent from training and map them to an explicit unseen class for scoring. The models have no rejection or unknown-class detector. Known-class-only Macro-F1, worst-well and W10 are reported separately, so excluding the forced-zero unseen class is not mistaken for open-set recognition. Historical seeds~42, 43 and~45 are closed-set, with 0\% unseen test depths. Seed~44 is an open-set stress: unseen facies occupy 4.4\% of test depths ($1.08\times10^4$ / $2.45\times10^5$), and 3899 validation rows are dropped because their labels are absent from training (Table~\ref{tab:regime}).

The primary aggregation uses the locked 20-seed list, with 18 closed and 2 open splits (Table~\ref{tab:multirep}). Means over seeds 42--45 are sensitivity summaries only. Among those four seeds, only seed~44 separates related NPD wellbores across training and testing (\texttt{34/5-1 S}$\to$train, \texttt{34/5-1 A}$\to$test; Section~\ref{sec:parent}). Validation exclusions on the closed seeds are small, at 0--141 rows. Long-tailed class counts favour Macro-F1 over accuracy (Fig.~\ref{fig:class}; \citep{Japkowicz2002Imbalance,Sokolova2009Metrics}). The held-out wells also differ in class proportions and mean curve values (Supplementary Figs.~S1--S2).''')
    paragraph('main.tex', 'Historical pipeline-r3 runs used CPU-only', r'''
Historical r3 experiments used Python~3.11 on Windows CPU workstations at Southwest Petroleum University, without GPUs. A typical host had 12 cores, 16 threads and $\geq$32\,GB RAM. LightGBM/sklearn used \texttt{n\_jobs=-1} where applicable. Median multi-rep runtime was $\approx$3--4\,min per method--seed with LightGBM~4.7; dependency versions are pinned in \texttt{requirements.lock.txt}.

LightGBM and XGBoost stopped on well-held-out validation multi-class log-loss, with \texttt{n\_estimators}=180. Global-RF used 200 trees under the $\leq$120k schedule and 100 under the equal-row full schedule, without early stopping. MLP used sklearn's internal $\approx$10\% row hold-out on accuracy, rather than well-level validation. Hyperparameters were shared across variants, with \texttt{class\_weight=balanced} and row bagging disabled (Section~\ref{app:impl}). RF and CORAL schedules are detailed in Appendices~\ref{app:rf} and~\ref{app:impl}; equal-row RF/XGBoost used the full Global-LGBM training cohort.

Historical ablations use Protocol-1 seeds~42/43/45 and Protocol-2/3 seed~42, as defined in Section~\ref{sec:claims}. The six Protocol-3 targets are 15/9-23, 17/11-1, 29/3-1, 33/9-1, 34/10-16~R and 7/1-1. An author-selected index-shifted draw has no r3 scores (Section~\ref{sec:lowo}). Configurations, CSV summaries, detail JSON and \texttt{RELEASE\_MANIFEST.json} accompany the code materials.''')
    paragraph('v2_methods.tex', 'The extension locks seeds', r'''
The restricted-curve pilot uses closed seeds 42, 43 and 45, with open seed 44 reported separately. Global, WellNorm, Hybrid-norel and Hybrid share source training/validation wells and labelled test depths within each seed. We compare GR, RHOB, NPHI and DTC with the original sixteen channels, retaining rows with missing values. The curve restriction therefore leaves the labelled cohort unchanged.

Training counts, class encoding and weights, source-fitted imputation and scaling, the 180-tree budget and validation criterion remain fixed. New fits use eight threads. We neither tune on test scores nor search additional models.''')
    paragraph('v2_methods.tex', 'An audit identified a feature-budget mismatch', r'''
The r3 audit found that WellNorm generated masks after normalisation had replaced missing values with zero. Its masks were consequently all zero. The corrected implementation records $m_c=\mathbb{1}\{x_c\text{ is missing}\}$ from the original observed channel. Global and Hybrid retained raw columns in r3, so their feature matrices are unchanged; fitting and inference tests check this equivalence.

We fit all sixteen four-curve method--seed combinations and four corrected sixteen-curve WellNorm models. Twelve stored sixteen-curve Global/Hybrid references retain their original provenance. A further seed-42 WellNorm fit keeps the legacy masks at eight threads to isolate the mask change from a thread-setting difference. The pilot therefore adds twenty fits and one diagnostic, rather than repeating the full twenty-split schedule.''')
    paragraph('v2_methods.tex', 'Reported uncertainty is the sample standard', r'''
Standard deviations describe sensitivity across the three overlapping closed splits, not population confidence intervals. Curve-set differences are paired by seed. Open seed 44 is scored separately without an unknown-class detector. Because this study retains missing observations and its own well splits, the comparison does not numerically reproduce the preprocessing of \citet{Carvalho2026LithologyBenchmark}.''')
    paragraph('v2_methods.tex', 'Saved predictions from all eighteen', r'''
Saved predictions from the eighteen closed splits in each grouping mode provide per-well present-class F1 differences for Hybrid-norel minus Global. We first average repeated appearances within a well, then average sibling wells within a parent. Frequently sampled wells and siblings thus contribute one parent-level point to descriptive associations. Shared models and source wells still induce dependence, so these points are not independent population replicates. Present-class F1 measures within-well predictive change; fixed-taxonomy tails retain their separate coverage-aware role.''')
    paragraph('v2_methods.tex', 'Observable descriptors include', r'''
Observable descriptors are full-curve missing fractions for four and sixteen channels, median $|\mathrm{CALI}-\mathrm{BS}|$, median $|\mathrm{DRHO}|$, and a four-curve location difference. The last averages absolute differences between target-well curve means and source-labelled medians, each divided by its source-labelled standard deviation. All can reflect geology, measurement and sampling together.

True class counts, dominance and source--target class-prior total variation require target labels and are used only post-hoc. We report exploratory Spearman associations and selected gain/loss examples without significance tests, fitted thresholds or causal attribution. The Supplementary mean-shift identity algebraically separates composition and within-class terms. It does not identify geological or instrument effects.''')
    paragraph('v3_methods.tex', 'After the initial four-curve pilot,', r'''
Before fitting the follow-up models, we fixed seeds 20261103, 20261104 and 20261105. Each partition keeps wellbores sharing a parent together in training, validation or testing. All seeds are retained. Closed/open membership follows source/test class coverage rather than model scores, and all three are closed. These new partitions still overlap in FORCE wells; they provide neither an external dataset nor independent population repetitions.''')
    paragraph('v3_methods.tex', 'For each partition, Hybrid and Hybrid-norel', r'''
Hybrid and Hybrid-norel are fitted under both curve sets in each partition, giving twelve fits. They retain the earlier full-row budget, raw masks, hyperparameters and source-well validation. The primary descriptive contrast is the paired difference of differences''')
    paragraph('v3_methods.tex', 'where $H$ is Hybrid and', r'''
Here $H$ denotes Hybrid and $HN$ Hybrid-norel. Terms within a split use identical scored depths. We report each $I_s$, its mean and split standard deviation without population inference. Before running the new schedule, a fitting gate reproduced five v2 four-curve Hybrid seed-42 metrics to $10^{-10}$. Boosters, transformation state and row-level predictions are saved.

The first gate fit failed during native Windows Unicode-path model saving. Python model-text writing and readback resolved the issue without changing the experimental schedule. The phase comprises thirteen successful fits and one gate attempt that failed at persistence.''')
    paragraph('v3_methods.tex', 'For each of the six fitted Hybrid models,', r'''
For each of the six fitted Hybrid models, the target well's full observed interval is $[d_{\min},d_{\max}]$, with $L=d_{\max}-d_{\min}$. The fixed stresses remove 10\% or 20\% of $L$ from the top, bottom or both. With fractions $a,b$, we retain labelled depths in $[d_{\min}+aL,d_{\max}-bL]$ and recalculate relative depth as''')
    paragraph('v3_methods.tex', 'Source fitting, raw curves, missingness masks', r'''
Only the relative-depth coordinate changes. Source fitting, raw curves, masks and whole-well $z$-scores remain frozen. For each crop, baseline Hybrid, changed-coordinate Hybrid and Hybrid-norel are scored on identical retained rows and class denominators. The within-crop stress effect is $F1_{H,r'}-F1_{H,r}$, so changes in retained class composition do not masquerade as coordinate effects.

All scenarios retain row, well and class counts alongside fixed-taxonomy tails. Inference reuses fitted models, without additional fits or test-label-based tuning. Since normalisation moments still use the full curve, this is a coordinate-sensitivity check, not a physical incomplete-log simulation.''')
    paragraph('v3_methods.tex', 'Boundary coordinates are reconstructed', r'''
Coordinates use original float64 dataframe depths, rather than the float32 depths exported in FeatureBundle. Reconstructed baseline coordinates match the trained features exactly. Before applying stresses, we also verify saved-model baseline predictions on every test row. This recovery adds no model fits.''')


def edit_results():
    paragraph('main.tex', 'To diagnose adjacent-depth leakage versus', r'''
Seed~42 Global-LGBM diagnostics distinguish protocol sensitivity from target-well and class-composition shifts (Table~\ref{tab:matched}). The authors selected these contrasts before final aggregation; companion scripts are provided.

Random depth-point splitting (A) gives Macro-F1 0.737, compared with 0.497 for contiguous within-well depth-block hold-out (B), a change of $-0.240$. Wells and sample counts are nearly the same, but test depths and class priors differ. The A$\to$B contrast therefore measures protocol sensitivity rather than a pure neighbour-leakage effect.

The Protocol-1 substitution diagnostic holds source training wells, validation wells and scored test depths fixed across $C^\star$, D and $D_{\mathrm{eq}}$. D adds other labelled depths from the target wells ($+1.10\times10^5$ rows). The equal-$n$ draw $D_{\mathrm{eq}}$ combines these with source rows and downsamples to $|C^\star|$. Macro-F1 changes from 0.436 for $C^\star$ to 0.464 for D ($+0.028$), or 0.459 for the published equal-$n$ draw ($+0.024$). Ten subsample seeds with fixed test blocks give mean $\Delta=+0.020\pm0.006$, ranging from $0.012$ to $0.031$.

Equal row counts do not isolate neighbour leakage. The draw replaces about 10\% of source rows with target-well labelled depths, changes class priors and scaling statistics, and adds the 17 target wells to labelled training. Full well hold-out C scores 0.404 and also differs in well/class composition from A/B. These diagnostics cannot be added to partition the 0.737$\to$$\approx$0.34 headline gap (Table~\ref{tab:multirep}).''')
    paragraph('main.tex', 'The primary scores are Macro-F1', r'''
We report pooled Macro-F1 over classes present in the evaluated ground truth, alongside fixed-taxonomy worst-well Macro-F1. The latter uses shared train-learnable labels within each split, excluding unk (\texttt{fixed\_label\_mode}=\allowbreak\texttt{train\_learnable}). Missing classes receive F1$=0$ (\texttt{zero\_division}=0), so every well in that split has the same denominator.

For a closed-set well with $k$ observed classes among $K$ train-learnable labels,
$F1_{\mathrm{fixed},w}=(k/K)\,F1_{\mathrm{present},w}$, and perfect predictions reach at most $k/K$. A perfectly classified single-facies well with $K=11$ therefore scores only $\approx0.091$. The identity does not directly extend to open-set wells whose present-class denominator includes excluded unk. Fixed-taxonomy tails combine class coverage and predictive quality; they are neither pure error rates nor calibrated failure probabilities.

For cross-split comparison, $C_{\mathrm{common}}$ is the intersection of train-learnable class names, with missing-class F1$=0$. CSVs also retain the legacy present-class minimum, \texttt{well\_true}, using each well's own $y_{\mathrm{true}}$. Those denominators vary, allowing single-class wells to inflate the minimum. On seeds~42/43/45, Global/Hybrid mean worst-well falls from about $0.20$--$0.22$ under \texttt{well\_true} to $0.10$--$0.11$ under fixed taxonomy (Table~\ref{tab:multiseed}; \texttt{fixed\_taxonomy\_protocol1.csv}). Supplementary Table~S9 compares $k$, $K$, both F1 measures and accuracy. Fixed and present-class minima select different wells in 8/18 closed Global repetitions.

W10 averages the worst 10\% of wells under fixed taxonomy. With $n_{\mathrm{test\ wells}}=17$, it includes $\lceil 0.1\times 17\rceil=2$ wells and is less brittle than a single minimum. Pooled Macro-F1 includes never-predicted unk when it appears in test truth, as on seed~44. Historical $n=3$ seed-delta bootstrap intervals and win/loss counts describe ablations (42/43/45). Primary paired $\Delta$/wins use $n=18$ overlapping closed repetitions and describe split sensitivity, not population uncertainty (Table~\ref{tab:pairedmr}). Accuracy and weighted F1 remain secondary under the skewed class distribution (Fig.~\ref{fig:class}; \citep{Japkowicz2002Imbalance}). The FORCE penalty matrix is discussed qualitatively in Section~\ref{sec:forcecmp}.''')
    paragraph('main.tex', r'\paragraph{Legacy WellNorm qualification.}', r'''
\paragraph{Historical WellNorm results.} In r3, WellNorm and WellNorm-norel computed masks after replacing missing values with zero. Their historical scores, including the Protocol-2/3 tables, cannot isolate the effect of retaining global levels or establish that normalisation itself is harmful. Global and Hybrid features are unchanged by the mask correction, preserving their 18-split headline comparison. Corrected WellNorm evidence is limited to the matched pilot (Section~\ref{sec:v2pilot}).''')
    paragraph('main.tex', r'\revmark{Following the evidence spine', r'''
The primary results use the locked Protocol-1 well-id aggregation over $n=18$ closed splits, excluding open seeds~44 and~127. HN denotes Hybrid-norel and WNn denotes WellNorm-norel. Worst-fixed and W10-fixed use the fixed taxonomy; Worst-true is the legacy present-class minimum (\texttt{well\_true}). Seed~42 and historical seeds~42/43/45 provide diagnostics and ablations, not alternative headline means (Section~\ref{sec:claims}).''')
    paragraph('main.tex', r'\revmark{\emph{Mean performance.}', r'''
\paragraph{Mean performance.} Global-LGBM reaches Macro-F1 $0.339\pm0.045$, while Hybrid-norel, Hybrid and Hybrid-while lie near $0.370$--$0.379$ (Table~\ref{tab:multirep}). The table reports means and sample standard deviations across the 18 overlapping closed splits, alongside $C_{\mathrm{common}}$, Worst-fixed and W10-fixed. These deviations describe split sensitivity rather than population standard errors.

The common-class measure covers 11 shared train-learnable facies. A shared class absent from test support receives F1$=0$; support varies for Halite (10/18), Anhydrite (12/18) and Chalk (15/18). Restricting $C_{\mathrm{common}}$ to classes with positive test support gives scores close to pooled Macro-F1.''')
    paragraph('main.tex', r'\revmark{\emph{Tail trade-off and method choice.}', r'''
\paragraph{Well-level trade-offs.} Higher mean Macro-F1 does not consistently improve the worst-well score (Fig.~\ref{fig:pareto}). On well-id splits, Hybrid exceeds Global (0.379 versus 0.339) and HN (0.370) in mean Macro-F1. Its Worst-fixed is lower than HN's, at 0.101 versus 0.105; Global scores 0.093. HN improves both summaries relative to Global, with paired mean $\Delta$ Macro $+0.032$ and 15/18 wins (Table~\ref{tab:pairedmr}). Those counts describe overlapping splits, not a population test.

The parent-grouped companion changes this trade-off (Table~\ref{tab:multirepparent}): Hybrid$-$Hybrid-norel mean Worst-fixed becomes positive, at $+0.006$ with 13/18 wins. Because fixed-taxonomy tails also reflect coverage, neither this change nor a single minimum establishes deployment safety (Section Metrics; Supplementary Table~S9).''')
    paragraph('main.tex', r'\revmark{\emph{Sensitivity checks and historical ablation.}', r'''
\paragraph{Sensitivity checks.} Including all 20 well-id splits, with open seeds~44 and~127, gives Macro-F1 $0.337\pm0.043$ for Global, $0.360\pm0.052$ for HN and $0.370\pm0.044$ for Hybrid. Their Worst-fixed means are 0.090, 0.102 and 0.100. Excluding seed~42 leaves $n=17$ closed splits, with Macro-F1 $0.335\pm0.043$, $0.368\pm0.045$ and $0.377\pm0.038$, respectively. The HN--Global paired difference remains $+0.034$, with 15/17 wins.

The three-seed closed comparison (42/43/45; Table~\ref{tab:multiseed}; Fig.~\ref{fig:multiseed}) and historical $2\times2$ table (Supplementary Table~S17) remain ablation evidence. Equal-row RF-full scores 0.402 on those three seeds, above Global-LGBM's 0.371, so the protocol effect is not an argument for LightGBM superiority. Machine-readable summaries and the filled reporting record accompany these results (Tables~\ref{tab:checklist}--\ref{tab:accept}).''')
    paragraph('main.tex', 'Under the naive depth-point regime', r'''
Global-LGBM reaches Macro-F1 0.737 under the naive seed~42 depth split, compared with $0.339\pm0.045$ across the primary closed well-held-out repetitions (Table~\ref{tab:multirep}; Fig.~\ref{fig:leak}). The historical three-seed value, $\approx$0.371 on 42/43/45, is an ablation summary; the four-seed mean including open seed~44, $\approx$0.36, is a sensitivity check (Table~\ref{tab:multiseed}).

Table~\ref{tab:matched} helps interpret the 0.737$\to$$\approx$0.34 gap. A$\to$B changes Macro-F1 by $-0.240$ without changing well identity, although test depths and priors differ. D$-C^\star$ adds target-well non-test depths and gives $+0.028$. Equal-budget substitution $D_{\mathrm{eq}}-C^\star$ gives $+0.024$, with ten-subsample mean $+0.020\pm0.006$. Neither contrast isolates neighbour leakage alone. C also changes well and class composition relative to A/B (Supplementary Figs.~S1--S2). The diagnostics cannot be summed into causal components; Figure~\ref{fig:leak} describes combined protocol/deployment optimism.''')
    paragraph('main.tex', 'A fixed-test-block embargo inside', r'''
The seed~42 fixed-test-block embargo retains disjoint training/validation wells and ten equal-$n$ subsample seeds (pipeline~\texttt{r3}). Mean Macro-F1 changes from $0.425\pm0.006$ at $0$\,m to $0.410\pm0.004$ at $20$\,m, remaining near $0.42$. Legacy Worst-true declines overall from $0.146$ to $0.105$, though intermediate distances are not monotonic. Class-prior TV stays near $\approx$0.145 (Supplementary Table~S6). Together with Table~\ref{tab:matched}, this supports reading the 0.737$\to$$\approx$0.34 gap as combined protocol optimism, rather than a pure leakage penalty (Table~\ref{tab:multirep}).''')
    paragraph('main.tex', 'Whole-well $z$-scores and relative depth assume', r'''
Hybrid-while replaces completed-well moments with an expanding causal prefix and omits relative depth. This corpus presents a mild prefix test: on seed~42, $220733/220839$ labelled test depths have $n_{\mathrm{prefix}}\geq100$ (Supplementary Table~S1). The result concerns causal-prefix normalisation of wireline curves. It does not establish MWD/LWD deployment capability or performance in early-bit and sparse-prefix intervals.

Across the primary closed repetitions, Hybrid-while reaches Macro-F1 $0.378\pm0.041$ (Table~\ref{tab:multirep}). The historical three-seed comparison gives 0.412 versus Global's 0.371, with mean Worst-fixed 0.111 versus 0.104 (Table~\ref{tab:whilems}). Early-bit and sliding-window stresses appear in Supplementary Table~S8; their windows are not production defaults.''')
    paragraph('v2_results.tex', 'Table~\\ref{tab:v2core} reports', r'''
Restricting the pilot to four curves reduced mean Macro-F1 for all four methods (Table~\ref{tab:v2core}). Global, corrected WellNorm, HN and Hybrid scored $0.296\pm0.031$, $0.277\pm0.044$, $0.321\pm0.049$ and $0.364\pm0.044$. Their sixteen-curve counterparts scored $0.371\pm0.069$, $0.320\pm0.033$, $0.401\pm0.061$ and $0.409\pm0.059$.

Paired four-minus-sixteen changes were $-0.075$ for Global, $-0.042$ for WellNorm, $-0.080$ for HN and $-0.045$ for Hybrid. These three-split estimates describe input sensitivity, not an eighteen-split result or a ranking against another preprocessing pipeline. Pooled and Worst-fixed changes also differed (Fig.~\ref{fig:v2core}).''')
    paragraph('v2_results.tex', 'Within the four-curve setting,', r'''
Adding relative depth to HN increased mean Macro-F1 by $+0.043$ with four curves and $+0.008$ with sixteen. The paired difference of differences was $+0.035$. This pattern is descriptive: selected splits, correlated features and logged intervals preclude a general geological explanation or universal benefit. Open seed~44 is reported separately in Supplementary Table~S11.''')
    paragraph('v2_results.tex', r'\paragraph{Missingness-mask correction.}', r'''
\paragraph{Missingness-mask correction.} Corrected sixteen-curve WellNorm scores $0.356$ on seed 42, compared with the historical $0.293$. At eight threads, the legacy-mask fit reproduces saved Macro-F1 and Worst-fixed to $10^{-10}$; restoring raw masks changes Macro-F1 by $+0.063$.

The correction is not a consistent performance gain. Across the three closed pilot splits, new-minus-archived Macro-F1 averages $+0.014$, ranging from $-0.013$ to $+0.063$, with only one positive split. The same-thread diagnostic covers seed 42 only (Supplementary Table~S12). Its purpose is to restore comparable raw information. The historical Hybrid--WellNorm gap cannot be attributed entirely to global levels, and the corrected pilot does not replace the full WellNorm schedule or establish an intrinsically superior normalisation strategy.''')
    paragraph('v2_results.tex', 'After averaging repeated test appearances', r'''
The positive average gain conceals losses on individual wells (Fig.~\ref{fig:v2heterogeneity}). After averaging test appearances within wells and parents, HN exceeds Global in present-class F1 for 75/107 parents under well-id splitting. The mean paired change is $+0.027$, the median $+0.024$, and the 10th--90th percentiles $-0.043$ to $+0.100$. The parent-grouped companion has 78/106 positive effects, with mean $+0.027$ and median $+0.017$. Equal parent weighting makes these summaries distinct from pooled split Macro-F1.''')
    paragraph('v2_results.tex', 'Simple descriptors provide limited explanation.', r'''
The descriptors explain little of this variation. Under well-id splitting, Spearman associations with the paired gain are $-0.150$ for four-curve location difference, $+0.109$ for four-curve missingness and $-0.078$ for sixteen-curve missingness. Median $|\mathrm{CALI}-\mathrm{BS}|$ gives $+0.034$, and median $|\mathrm{DRHO}|$ gives $+0.178$, with 73--107 parents available across descriptors. True class-prior total variation gives $-0.030$ and requires target labels. These exploratory associations do not supply an advance-warning or model-selection rule.

The label-dependent decomposition is similarly sensitive to grouping. Composition-magnitude associations are $-0.086$ and $-0.160$, and within-class associations are $-0.245$ and $-0.142$, for well-id and parent-grouped schedules, respectively (Supplementary Fig.~S10). The decomposition does not causally separate calibration drift from geological change.''')
    paragraph('v3_results.tex', 'On the three new parent-grouped partitions,', r'''
The new parent-grouped partitions did not reproduce a uniform curve-set pattern. Relative depth changes mean Macro-F1 by $+0.019$ with four curves and $+0.008$ with sixteen. The mean paired difference of differences is $+0.011$, ranging from $-0.018$ to $+0.045$, and is positive in 2/3 partitions (Supplementary Table~S14).

Table~\ref{tab:v3new} gives the new baseline scores; Figure~\ref{fig:v3new} separates this follow-up from the initial pilot. These checks reduce reliance on the original three splits. Reuse of the same corpus and overlapping wells still prevents external-validation or population claims.''')
    paragraph('v3_results.tex', 'Across the six prespecified crops,', r'''
Changing the boundary coordinate reduced four-curve mean Macro-F1 in every prespecified crop, with changes from $-0.052$ to $-0.011$. Sixteen-curve mean changes ranged from $-0.024$ to $-0.001$ (Fig.~\ref{fig:v3boundary}). Retained rows varied from 59.1\% to 98.9\% across split--crop comparisons, and 30/36 model--crop changes were negative.

Each change compares Hybrid predictions on identical retained depths. Crop-related differences in class coverage therefore cannot alone explain the within-crop effect. Supplementary Tables~S15--S16 report all seeds, crops and matched-support HN comparisons. These results measure coordinate sensitivity, not the removal of physical measurements or recalculation of normalisation from partial logs. No crop was selected as a tuned inference policy.''')
    paragraph('v3_results.tex', 'On the matched four-curve support', r'''
Some coordinate changes reversed the average advantage over HN. On the shared four-curve support for a 20\% bottom crop, the mean original gain was $+0.035$, becoming $-0.017$ with the new coordinate. For a two-sided 20\% crop, it changed from $+0.034$ to $-0.018$. These reversals define an interval-sensitive boundary for this learner; they establish neither a universal ranking nor a geological mechanism.''')


def edit_ending():
    paragraph('main.tex', 'We use standard DG/UDA vocabulary', r'''
Following standard DG/UDA terminology, inductive DG uses source data alone, whereas UDA-lite may use unlabelled target curves \citep{Pan2010Transfer,BenDavid2010Theory,Zhou2023DGSurvey,Wilson2020UDASurvey}. Moment matching and adversarial adaptation motivate the CORAL appendix baseline \citep{Sun2016CORALAAAI,Ganin2016DANN}; GroupDRO motivates attention to well-grouped risk \citep{Sagawa2020GroupDRO}. Protocol-sensitive benchmarks provide the broader rationale for giving evaluation design comparable attention to model choice \citep{Koh2021WILDS,Gulrajani2021DomainBed}.''')
    paragraph('main.tex', 'Protocol-1 splits by wellbore identity.', r'''
Among seeds~42--45, only seed~44 separates the related \texttt{34/5-1} wellbores across training and testing, despite their $\approx$1.33\,km surface separation. The prespecified parent-grouped companion assigns related wellbores together across the same locked schedule, with 18 closed and 2 open splits. Global's grouped Macro-F1, $0.340\pm0.047$, remains close to the well-id value, $0.339\pm0.045$ (Table~\ref{tab:multirepparent}). The historical four-seed Global comparison is retained in Supplementary Table~S7.

Removing sibling \texttt{34/5-1 S} from seed~44 training changes Macro-F1 from $0.326$ to $0.324$ and legacy Worst-true from $0.082$ to $0.093$. Sibling overlap is unlikely to dominate that seed's score, though the collision remains part of the reported split definition.''')
    paragraph('main.tex', r'\emph{Within} test-domain normalisation', r'''
The seed~42 results illustrate the historical feature comparisons (Table~\ref{tab:p1}). HN scores 0.404/0.130/0.141 for Macro-F1, Worst-fixed and W10, versus legacy WellNorm's 0.293/0.105/0.120; Hybrid scores 0.411/0.118/0.143. These are not isolated global-level effects: HN$-$WellNorm changes two nominal features, and WellNorm also loses raw masks (Supplementary Table~S17).

Equal-row Global-LGBM ($\leq$120k) gives 0.403, close to the full model. Unequal-budget Global-RF scores 0.423/0.092; equal-row RF-full and XGBoost score 0.449/0.079 and 0.411/0.089. Source-only MLP gives 0.365/0.061, with CORAL-MLP retained in the appendix. Full-split methods all train on $9.86\times10^5$ rows on this seed.''')
    paragraph('main.tex', 'The default CORAL baseline is a shallow MLP', r'''
The appendix CORAL baseline uses an MLP with hidden sizes 96/48, $\leq$120k training rows, and sklearn early stopping on internal $\approx$10\% row-holdout accuracy. It does not use well-held-out validation. Alignment draws $\leq$40k target rows from labelled scoring positions with labels hidden (\texttt{uda\_target\_mode=transductive\_score\_support}), rather than from the full unlabelled $W_t$ curve. Source-only MLP uses the same inductive global features, without relative depth.

The implementation is covariance-only, source-scaler CORAL. A source-fitted \texttt{StandardScaler} transforms both domains; source covariance is then mapped towards target covariance at training time \citep{Sun2016CORALAAAI}. Inference applies the source scaler only, without reapplying map~$A$. This is not dual-centering CORAL.

On seed~42, CORAL-MLP scores Macro-F1 0.312 and Worst-fixed 0.033, versus 0.365/0.061 for source-only MLP. The stronger 256/128-hidden-unit variant uses up to 200k training rows and 80k target rows, yet remains below LightGBM. These unequal CPU budgets do not establish a general failure of moment matching relative to ERM trees. Equal-budget DANN-style adversarial adaptation remains a future GPU-oriented comparison.''')
    paragraph('main.tex', 'All reported Protocol-1 multi-rep timings', r'''
Historical Protocol-1 timings use CPU-only Windows workstations, typically 12 cores, 16 threads and $\geq$32\,GB RAM, without GPUs. LightGBM~4.7 and sklearn use \texttt{n\_jobs=-1}. Across $n=18$ closed well-id repetitions and six LightGBM methods, median runtime is 211\,s per fit (IQR 174--256\,s). Global-LGBM takes 181\,s (143--199), Hybrid 236\,s (179--253), and HN 239\,s (184--260). Parent-grouped fits have comparable runtimes. Individual timings remain in the CSV \texttt{seconds} field.''')
    paragraph('main.tex', 'Table~\\ref{tab:forcecmp} contrasts evaluation', r'''
Table~\ref{tab:forcecmp} compares the studies' emphasis rather than their scores, since metrics and splits differ. Algorithm comparisons \citep{Merembayev2021Energies,Mousavi2024GrowNet} and foundation-model accuracy reports \citep{Qi2025WLFM} ask different questions from our well-held-out reporting framework. Here a fixed LightGBM family exposes the change from 0.737 to $\approx$0.34 for Global, with Hybrid variants near 0.37--0.38. Information budgets and well tails determine how those scores should be read.

Equal-row RF and XGBoost are inductive companions, and RF-full can exceed Global-LGBM in mean Macro-F1 (Table~\ref{tab:multiseed}). They do not turn this study into an algorithm ranking. Unequal-budget CORAL-MLP remains in the appendix; the historical non-r3 FORCE penalty comparison is qualitative and outside the r3 result set.''')
    paragraph('main.tex', 'The results support a conditional interpretation', r'''
Feature gains depend on the evaluation setting. Post-drilling Hybrid improves mean performance over Global, yet well tails and parent-averaged effects show that the gain does not extend to every well. The initial four-curve pilot also suggested a larger benefit from relative depth. New parent-grouped partitions narrowed that difference to $+0.011$, with only two of three positive. This is not a universal curve-count effect. Since both partition and grouping change between phases, the difference cannot be attributed to parent grouping alone.''')
    paragraph('main.tex', 'The boundary-coordinate experiment provides', r'''
Boundary stresses give a more direct test within the fitted models. Changing relative depth alone reduces mean four-curve Macro-F1 in all six crops, and reverses the average HN comparison for bottom and two-sided 20\% crops. The learner is sensitive to the definition of the observed interval.

This sensitivity has a specific scope. Whole-well normalisation moments remain frozen, so the experiment establishes neither physical incomplete-log performance nor a geological cause. Relative depth should be treated as an interval-dependent post-drilling feature, with its logging support stated explicitly.''')
    paragraph('main.tex', 'The historical WellNorm comparisons also', r'''
The missingness audit imposes a separate limit on interpretation. Historical WellNorm comparisons lost raw masks and cannot isolate the contribution of global levels. Supplementary Tables~S17--S19 retain those diagnostics; corrected evidence comes from the matched pilot. Similarly, the causal-prefix results concern the observed completed-wireline coverage, not early-bit deployment. Together, these findings argue for reporting information access, preprocessing support and well tails alongside the score \citep{Koh2021WILDS,Zhou2023DGSurvey,Gulrajani2021DomainBed}.''')
    paragraph('main.tex', 'Table~\\ref{tab:accept} is the filled', r'''
The filled reporting record uses the $n=18$ closed well-id results (Table~\ref{tab:accept}; machine-readable version in \path{results/followup/}). The parent-grouped companion is Table~\ref{tab:multirepparent}. A complete record states the information budget and training count, then reports Macro-F1 with Worst-fixed and W10. Here W10 contains $\lceil 0.1\times n_{\mathrm{test\ wells}}\rceil$, or 2 of 17 wells. These tails are coverage-aware screens, not calibrated failure probabilities. ``Acceptance'' refers to reporting completeness, not field certification; depth-point scores alone do not establish performance on new wells.

Global-LGBM fits settings where target aggregates are unavailable. Hybrid-while is an option when completed-wireline causal prefixes are admissible and coverage resembles this corpus ($n_{\mathrm{prefix}}\geq100$ at nearly all scored depths; Supplementary Table~S1). FORCE channels are not uniformly real-time MWD/LWD measurements, leaving early-bit and sparse-prefix performance unresolved.

For post-drilling interpretation, available curves, the corrected pilot and the Pareto comparisons help assess Hybrid-norel and Hybrid. Higher well-id Hybrid Macro-F1 need not improve Worst-fixed, and the parent-grouped Hybrid$-$HN tail difference can change sign (Table~\ref{tab:pairedmr}). Centred smoothing remains offline, with window~7 fixed in advance. Macro-F1 near 0.34--0.38 and rare-facies failures support an interpreter-assistance role. Review heuristics in Section~\ref{sec:rare} remain bounded by Section~\ref{sec:limitations}.''')
    paragraph('main.tex', 'Primary closed multi-rep uses', r'''
The $n=18$ Protocol-1 repetitions overlap, so their standard deviations and paired changes measure split sensitivity rather than population uncertainty. Removing the ablation seed~42 leaves 17 closed repetitions: Global, HN and Hybrid change from 0.339/0.370/0.379 to 0.335/0.368/0.377. Including all 20 splits gives 0.337/0.360/0.370. The ordering is stable, but repeated use of wells still limits inference. Population estimates would require nested well-level or hierarchical resampling. The seed list was fixed before final provenance capture, without formal preregistration.''')
    paragraph('main.tex', 'The remaining limitations concern deployment scope.', r'''
Unseen facies occur on open seeds~44 and~127, and the classifier has neither an unknown-class detector nor calibrated rejection. Rare-facies review heuristics therefore provide no selective-prediction guarantee (Section~\ref{sec:rare}). Fixed-taxonomy tails also combine coverage and prediction quality rather than estimating deployment-failure probabilities.

Protocols~2--3 are fixed-draw stresses, not full leave-one-well-out validation. Hybrid-while uses causal prefixes from completed wireline curves, not a complete measurement-while-drilling system. Equal-row RF/XGBoost comparisons do not match model capacity; strong UDA and confidence-weighted training were not assessed. The $0.737\to0.34$ contrast combines protocol and domain effects rather than identifying a pure leakage penalty.

The initial four-curve pilot has three overlapping closed splits and one open stress, rather than 18 new repetitions. Retaining missing rows and our own well splits distinguishes it from the processing and CV of \citet{Carvalho2026LithologyBenchmark}. New models use eight threads; reused references used \texttt{n\_jobs=-1}, so runtimes are not compared. Corrected WellNorm evidence covers only this pilot, leaving its historical 18-split mean as a legacy result. Descriptor and class-conditioned shift analyses are exploratory, and label-dependent quantities cannot guide deployment. Independent-corpus validation, validated review rules and population-level inference remain open.''')
    paragraph('main.tex', 'The follow-up adds three parent-grouped partitions', r'''
The follow-up adds three parent-grouped FORCE partitions, not independent external data or population repetitions. Its coordinate stresses retain full-curve $z$-scores and do not simulate the whole incomplete-logging workflow. No validated rule yet selects a method for a new well.''')
    paragraph('main.tex', 'A fixed LightGBM family on FORCE~2020', r'''
Cross-well lithofacies scores need both an information budget and a well-level assessment. With a fixed LightGBM family on FORCE~2020, inductive Macro-F1 is $0.339\pm0.045$ across 18 closed well-held-out repetitions, versus 0.737 under a naive depth split. Post-drilling Hybrid variants reach $0.370$--$0.379$, without consistent gains in coverage-aware well tails. The optimism contrast combines protocol and domain-composition changes; it does not isolate leakage causally.''')
    paragraph('main.tex', 'The restricted-curve pilot and three new', r'''
Relative-depth gains also vary. Across three new parent-grouped partitions, the four-minus-sixteen-curve difference of gains averages $+0.011$, with two of three positive. On identical retained depths, boundary-coordinate stresses can reverse the mean four-curve advantage over HN. These results define input- and interval-dependent benefits within the evaluated representations, rather than universal model superiority. The mask audit qualifies historical WellNorm comparisons, while simple well descriptors still offer no validated selection rule.''')
    paragraph('main.tex', 'The contribution is a reproducible framework', r'''
The framework links well-held-out estimation to information access, log availability, interval definitions and coverage-aware tails. The models remain interpretation aids. Independent-corpus validation, calibrated rejection and population-level inference are still needed.''')
    paragraph('main.tex', 'Unless stated otherwise we use', r'''
The default LightGBM settings are \texttt{n\_estimators}=180, \texttt{learning\_rate}=0.05, \texttt{num\_leaves}=63, \texttt{min\_child\_samples}=50, \texttt{colsample\_bytree}=0.8, \texttt{reg\_lambda}=1.0 and \texttt{class\_weight=balanced}. Early stopping uses well-held-out validation multi-class log-loss with patience~30. Training retains the full protocol split, about $0.97$--$1.02\times10^6$ rows under Protocol-1.

Row bagging is disabled (\texttt{subsample}=1.0, \texttt{bagging\_freq}/\texttt{subsample\_freq}=0). An earlier configuration specified \texttt{subsample}=0.8 without a positive bagging frequency, which leaves row sampling inactive. Reported results follow that no-bagging schedule.''')


def edit_supplement():
    paragraph('supplementary.tex', 'Worst-true (legacy) Macro-F1 on this diagnostic curve', r'''
Worst-true on this diagnostic curve is the per-well present-class minimum (\texttt{well\_true}). The main text instead uses Worst-fixed over shared train-learnable labels. LightGBM retains $n_{\mathrm{estimators}}=180$, matching Protocol-1. Since the curve scores the seed~42 test partition, it supports neither nested model selection nor the choice of the released window~7 default. Fixed-taxonomy results remain in \texttt{results/followup/fixed\_taxonomy\_protocol1.csv}.''')
    paragraph('supplementary.tex', 'Protocol-1 seed~42 Hybrid-while predictions stratified', r'''
We stratify seed~42 Hybrid-while predictions by the full-curve prefix preceding each labelled test depth (\texttt{scripts/while\_coldstart\_study.py}). Nearly every depth already has $\geq$100 prefix samples. In the large $n_{\mathrm{prefix}}\geq500$ bin, Macro-F1 is 0.434 and legacy Worst-true is 0.202, close to the whole-test score. The corpus therefore tests mild causal-prefix normalisation, rather than sparse-prefix or early-bit conditions.''')
    paragraph('supplementary.tex', 'The six wells below are selected post-hoc', r'''
The six examples comprise the three largest gains and losses among wells appearing in at least two closed well-id test splits. Selection is post-hoc. Each effect averages Hybrid-norel minus Global present-class F1 across appearances, illustrating heterogeneity rather than a validated selection rule.''')
    paragraph('supplementary.tex', 'For a curve observed in both source and target,', r'''
For a curve observed in both domains, $q_{s,k}$ and $q_{t,k}$ are class proportions on non-missing labelled samples. The corresponding class means are $\mu_{s,k}$ and $\mu_{t,k}$. Their observed mean difference satisfies''')
    paragraph('supplementary.tex', 'The first term is the composition term relative', r'''
The first term captures composition differences relative to source class means; the second captures within-class differences. Both are divided by the observed source standard deviation. We average absolute magnitudes across available core curves, then average repeated appearances within wells and parents. A curve is omitted when a target-observed class has no observed source mean.

The identity holds across 2460 curve records, with maximum source-standardised residual $5.91\times10^{-15}$. It is an algebraic description using true target labels. Within-class differences can arise from fluids, geology, measurement, sampling or labelling; they do not identify tool calibration effects.''')
    paragraph('supplementary.tex', 'All three new seeds are retained and closed.', r'''
All three new seeds are retained and closed, using new partitions of the same corpus rather than external data. Each boundary scenario scores changed Hybrid, baseline Hybrid and HN on identical depths and class denominators. Delta denotes changed-coordinate minus baseline Hybrid. Original and stressed gains compare Hybrid with HN before and after the coordinate change. All other features remain fixed.''')


def align_titles_and_layout():
    title = 'Cross-well lithofacies classification under restricted information and log availability: a well-held-out FORCE~2020 study'
    path = DST / 'main.tex'
    text = path.read_text(encoding='utf-8')
    text = text.replace('pdftitle={Information-budget-aware validation of cross-well lithofacies classifiers under domain shift}',
                        'pdftitle={' + title.replace('~', ' ') + '}')
    path.write_text(text, encoding='utf-8')
    path = DST / 'supplementary.tex'
    text = path.read_text(encoding='utf-8')
    text = text.replace('Supplementary material: Information-budget-aware validation of cross-well lithofacies classifiers under domain shift',
                        'Supplementary material: ' + title)
    # Keep the diagnostic heading with its following figure.
    text = text.replace(r'\section*{Hyper-parameter sensitivity (post-hoc diagnostics)}',
                        '\\clearpage\n' + r'\section*{Hyper-parameter sensitivity (post-hoc diagnostics)}')
    path.write_text(text, encoding='utf-8')


def audit():
    def tables(t):
        return re.findall(r'\\begin\{table\}.*?\\end\{table\}', t, re.S)
    def equations(t):
        return re.findall(r'(?<!\\)\\\[.*?(?<!\\)\\\]', t, re.S)
    def citations(t):
        return set(k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}', t) for k in block.split(','))
    def words(t):
        t = re.sub(r'\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}', '', t, flags=re.S)
        t = re.sub(r'\\(?:cite\w*|ref|label|path|url)\{[^}]*\}', '', t)
        t = re.sub(r'\$[^$]*\$|\\\[.*?\\\]', '', t, flags=re.S)
        return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", t))
    checks = {}
    before = after = ''
    for p in SRC.glob('*.tex'):
        old = p.read_text(encoding='utf-8'); new = (DST / p.name).read_text(encoding='utf-8')
        checks[p.name] = {'table_environments_identical': tables(old) == tables(new),
                         'display_equations_identical': equations(old) == equations(new),
                         'citation_keys_identical': citations(old) == citations(new)}
        if not all(checks[p.name].values()):
            raise AssertionError((p.name, checks[p.name]))
        if p.name in ['main.tex', 'v2_methods.tex', 'v2_results.tex', 'v3_methods.tex', 'v3_results.tex']:
            before += old + '\n'; after += new + '\n'
    for name in ['refs.bib', 'supplementary-refs.bib', 'table_v2_common_core.tex', 'table_v3_new_splits.tex']:
        if (SRC/name).read_bytes() != (DST/name).read_bytes():
            raise AssertionError(f'Protected file changed: {name}')
    # New numerical literals must already occur in the source manuscript.
    number = r'(?<![A-Za-z])\d+(?:\.\d+)?'
    introduced = sorted(set(re.findall(number, after)) - set(re.findall(number, before)))
    if introduced:
        raise AssertionError(('New numerical literals', introduced))
    report = {'edited_passages': len(EDITS), 'checks': checks,
              'main_prose_word_count_before': words(before), 'main_prose_word_count_after': words(after),
              'introduced_numerical_literals': introduced, 'new_experiments': 0,
              'supplement_title_and_pdf_metadata_aligned': True,
              'word_count_scope': 'English token count excluding table/figure blocks, math, citation and cross-reference fields; preamble and technical labels remain.'}
    (DST / 'LANGUAGE_EDITS.json').write_text(json.dumps(EDITS, indent=2, ensure_ascii=False), encoding='utf-8')
    (DST / 'LANGUAGE_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
