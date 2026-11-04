"""Build a shorter manuscript while preserving every existing numerical table."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse_evidence_20261104'
DST = ROOT / 'paper/gse_compact_20261104'


INTRO = r'''
\section{Introduction}\label{sec:intro}

Wireline logs constrain lithology between cores and seismic volumes \citep{Asquith2004Basic,Rider2011WellLogs}. Statistical classifiers, neural networks and tree ensembles have all been used to turn these measurements into facies predictions \citep{Busch1987Lithology,Rogers1992NN,Hall2016Facies}. Public datasets now make these methods easier to compare. FORCE~2020, in particular, provides labelled Norwegian wells with heterogeneous log coverage and lithofacies support \citep{Bormann2020FORCE}. The practical question is whether a score obtained from those data describes performance on a different well.

That question starts with the split. Random depth-point splitting shares well identity between training and testing; neighbouring depths also share borehole conditions and stratigraphic structure. Grouped cross-validation addresses this dependence in spatial and hierarchical data \citep{Roberts2017CV}. In our FORCE diagnostic, a fixed Global-LGBM classifier scores Macro-F1 0.737 under random depth splitting and 0.497 under contiguous depth-block hold-out (Supplementary Table~\ref{SI-tab:matched}). The primary well-held-out estimate is lower again, at $0.339\pm0.045$ across 18 closed partitions (Table~\ref{tab:multirep}). These contrasts change different aspects of the evaluation and require separate interpretation.

Holding out wells still leaves an information-access decision. An inductive classifier uses source-fitted parameters alone. Post-drilling normalisation also uses the target well's complete observed curves, whereas causal-prefix normalisation uses only samples up to the current depth. Domain-generalization and adaptation research distinguishes these forms of access \citep{Pan2010Transfer,Zhou2023DGSurvey,Wang2023DGSurvey,Wilson2020UDASurvey}; cross-well studies already address source reweighting and interpretation while drilling \citep{Chang2021GRSL,Xu2022JNGSE}. A score without its information budget obscures the difference between these settings.

Log availability adds another constraint. \citet{Carvalho2026LithologyBenchmark} benchmark FORCE and Geolink using four core curves and well-level five-fold cross-validation. We ask how that restricted curve set changes feature gains when the labelled rows stay fixed. We then test relative depth under new parent-grouped partitions and controlled changes to the observed interval coordinate (Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}). This design connects a representation's measured benefit to the information used to construct it.

We propose a FORCE evaluation framework that combines explicit information budgets, well hold-out and coverage-aware well tails (Fig.~\ref{fig:method}). A fixed LightGBM family supports the representation comparisons. Across the primary closed splits, Hybrid achieves mean Macro-F1 0.379 versus 0.339 for Global-LGBM, while the well-tail comparison depends on the representation and grouping rule (Tables~\ref{tab:multirep}--\ref{tab:pairedmr}). The framework therefore reports pooled and well-level results together.

The study makes three contributions. First, it implements an information-budget-aware comparison with a machine-readable reporting record (Table~\ref{tab:methods}; Supplementary Tables~\ref{SI-tab:checklist} and~\ref{SI-tab:accept}). Second, it quantifies pooled gains and fixed-taxonomy tails across 18 closed splits, with a parent-grouped companion (Tables~\ref{tab:multirep}--\ref{tab:pairedmr}). Third, it tests input and interval dependence through matched four- versus sixteen-curve experiments and inference-only coordinate stresses (Table~\ref{tab:v2core}; Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}). Saved predictions and execution manifests connect these findings to the released calculations.

\section{Related work}\label{sec:related}

Early lithology classifiers used statistical discrimination and back-propagation networks \citep{Busch1987Lithology,Rogers1992NN}. Later comparisons examined Bayesian, fuzzy, nearest-neighbour and neural approaches \citep{Dubois2007Facies}, followed by support vector machines for lithology and lithofacies in heterogeneous reservoirs \citep{AlAnazi2010Lithology,AlAnazi2010Lithofacies}. These studies establish a long methodological history. Our contribution concerns the evidence attached to cross-well scores rather than a new classification architecture.

Tree ensembles provide a useful fixed backbone for tabular representations. Random forests, gradient boosting and XGBoost supply the broader algorithmic context \citep{Breiman2001RF,Friedman2001GBM,Chen2016XGBoost}; LightGBM provides the implementation used for our principal comparisons \citep{Ke2017LightGBM}. FORCE-oriented work compares learning algorithms on Norway and Kazakhstan datasets \citep{Merembayev2021Energies} and combines neural boosting with transformed tabular inputs \citep{Mousavi2024GrowNet}. Convolutional methods also classify facies from logs \citep{Imamverdiyev2019CNN} and lithology from core CT images \citep{Chawshin2021CT}. These applications differ in data support, metrics and input modality, which prevents direct score ranking.

\label{sec:related-crosswell}
Cross-well transfer explicitly addresses distribution changes. \citet{Liu2020WellDrift} study lithology identification under logging-data drift. \citet{Chang2021GRSL} combine active learning with source reweighting, while \citet{Xu2022JNGSE} develop domain generalization for interpretation while drilling. Meta-learning \citep{Xu2022ICAICA}, adversarial and wavelet-based adaptation \citep{Sun2023Energies,Sun2023ITNEC}, sequential transfer \citep{Zhou2021MST}, partial adaptation \citep{Li2024TAI} and heterogeneous adaptation \citep{Ren2024SPE} further distinguish the available labels and domain assumptions. We use these studies to position information access, without treating their regional scores as FORCE baselines.

Well-held-out prediction is already established in this literature. For example, \citet{Chen2025FrontiersLithology} evaluate geological-information-driven deep learning on separate training and test wells. \citet{Carvalho2026LithologyBenchmark} provide the closest public multi-model benchmark: four logs, non-missing sequences and five-fold well-level validation on FORCE and Geolink. Our matched-row pilot retains missing observations and uses its own fixed well splits. It evaluates curve availability within our pipeline, rather than reproducing that benchmark's scores. The WLFM preprint reports lithology accuracy and few-shot adaptation \citep{Qi2025WLFM}; those outcomes also differ from pooled Macro-F1 and coverage-aware tails.

Broader machine-learning work explains why the reporting unit matters. Transfer-learning theory formalises source--target differences \citep{BenDavid2010Theory}; DG benchmarks show sensitivity to experimental and selection protocols \citep{Koh2021WILDS,Gulrajani2021DomainBed}. GroupDRO motivates attention to group-level risk \citep{Sagawa2020GroupDRO}, while class-imbalance and metric studies motivate class-balanced reporting \citep{Japkowicz2002Imbalance,Sokolova2009Metrics}. Our fixed-taxonomy tails retain a common within-split denominator across wells. Their interpretation depends on both predictive error and the classes present in each well, as defined below.
'''

METHODS = r'''
\section{Data and methods}\label{sec:method}\label{sec:problem}

\subsection{Data and supervised support}

We parse the FORCE~2020 LAS release into full-curve and labelled views \citep{Bormann2020FORCE}. The full view contains about $2.34\times10^6$ depths from 118 wells; about $1.43\times10^6$ carry lithofacies labels. Label coverage is about 61\% overall and ranges from about 9\% to 100\% across wells (Supplementary Table~\ref{SI-tab:coverage}). We retain confidence fields for traceability. The reported LightGBM fits use balanced class weights and do not weight samples by interpreter confidence.

The sixteen-channel representation uses GR, RHOB, NPHI, DTC, DTS, PEF, CALI, RDEP, RMED, RSHA, RXO, SP, ROP, DRHO, BS and MUDWEIGHT where present. The restricted representation uses GR, RHOB, NPHI and DTC. We retain the same labelled rows when restricting the curves, including rows with missing measurements. Whole-well and prefix statistics use all observable curve depths for the relevant well, independently of which depths have labels. Supervised fitting, early stopping and scoring use only the labelled view. Supplementary Fig.~\ref{SI-fig:class} gives the long-tailed class distribution.

Each labelled row contains curves $x_i$, facies $y_i$, confidence $c_i$, well identity $w_i$ and measured depth $d_i$. Source training wells supply labels for fitting; distinct source validation wells supply the early-stopping criterion. Target wells supply the final scored rows. In the principal well-held-out comparisons, target facies do not enter feature construction, classifier fitting or tuning. We use their class coverage to assign splits to the prespecified closed/open reporting groups and then to evaluate predictions. This bookkeeping does not change a split's trained model. The target-row diagnostic below deliberately changes labelled training access.

The two data views serve distinct purposes. An unlabelled depth contributes an observed curve value to its well's normalisation context but contributes no supervised training example. A missing curve value remains missing until the defined feature and imputation steps. We preserve the labelled cohort across methods within a split rather than selecting a different complete-case subset for each feature set. That distinction is especially important for the four-curve experiment: its score change reflects the chosen channel restriction under the same label support, while the separate masks preserve which measurements were unavailable at each retained depth.

\subsection{Well-held-out protocols}\label{sec:protocols}

Protocol-1 assigns approximately 70\% of wellbores to training, 15\% to validation and 15\% to testing. We sort well ids, shuffle them with the split seed and allocate by wellbore count. The locked schedule contains 20 seeds. A partition enters the primary closed summary when every test facies is learnable from its source training labels; seeds~44 and~127 instead form open-set stresses. All 20 remain in the unconditional sensitivity summary. We retain unseen test facies as an explicit unk class for scoring, without an unknown-class detector. Validation rows with labels absent from training do not enter early stopping (Supplementary Table~\ref{SI-tab:regime}).

The parent-grouped companion keeps related wellbores together. We define a parent by stripping a trailing trajectory token from the Norwegian well name, then shuffle parent keys and expand them into wellbores. Under this rule, only \texttt{34/5-1 A} and \texttt{34/5-1 S} share a parent in the labelled corpus. The companion applies the same locked seed list and closed/open rule. It tests sensitivity to the grouping unit rather than adding new wells; repeated partitions still share source and target wells.

The training label map uses the sorted source facies codes. A test code outside that map receives one additional unk index, which the fitted classifier never predicts. We keep those rows in pooled evaluation and save the number of excluded validation rows separately. Thus a closed-set mean and an all-split mean differ in the evaluated class regime, not in whether unseen test rows were silently removed. The open-set companions report both the original support and known-class-only scores, allowing readers to identify exactly which denominator changed.

We use three diagnostic settings to interpret Protocol-1. A naive baseline splits pooled depths uniformly by row. A contiguous within-well depth-block contrast changes local separation while retaining well identity. A matched target-row diagnostic fixes source training wells, validation wells and test depths, then adds labelled non-test depths from target wells. Its equal-row variant replaces source rows with target rows and downsamples to the original training count. These comparisons change different aspects of dependence, class composition and scaling support. Their scores do not form additive components of the naive-to-well-held-out gap (Supplementary Table~\ref{SI-tab:matched}).

Protocols~2 and~3 supply secondary stresses. Protocol-2 holds out NPD quadrants~15 and~35 separately; the legacy CSV field \texttt{block} stores quadrant ids. Protocol-3 tests six automatically selected wells, sorted by quadrant and sample count before taking index-spaced entries, against the remaining labelled wells. It is a fixed six-well draw rather than full leave-one-well-out over 118 wells. The selected wells, source/target counts and scores remain in Supplementary Tables~\ref{SI-tab:p2}--\ref{SI-tab:p3}. They supplement the primary multi-split result.

\subsection{Representations and admissible information}

Figure~\ref{fig:method} separates three information budgets. Global-LGBM uses source-fitted global levels and missingness indicators. Hybrid-norel (HN) adds whole-well $z$-scores. Hybrid also adds relative depth. Hybrid-while replaces whole-well moments with expanding causal-prefix moments and omits relative depth (Table~\ref{tab:methods}). We fit the classifier on source labels in every case; the distinction concerns which target observations enter the feature calculation.

For curve $c$, the global feature is $G_c=(x_c-\mathrm{med}_{s,c})/(s_{s,c}+10^{-6})$. The median and population standard deviation use labelled training rows only, as implemented in \path{src/features.py}. Within-well features use $Z_{c,w}=(x_c-\mu_{c,w})/(\sigma_{c,w}+10^{-6})$. Post-drilling moments use the complete observed curve for that well, including unlabelled depths; causal-prefix moments use depth-ordered observations up to and including the current depth. A degenerate standard deviation uses a unit fallback. Missing within-well responses become zero, with the missingness indicator preserving their distinction from an observed mean response.

Hybrid's relative depth is $r=(d-d_{\min})/(d_{\max}-d_{\min}+10^{-6})$, where each well's full observed depth range supplies its own bounds. Global-LGBM has no well-wise moments or relative depth. Hybrid-while has no total-interval coordinate. These definitions place Hybrid and HN in a post-drilling setting and Hybrid-while in a completed-wireline causal-prefix setting. They do not assume that all FORCE channels are available as measurements while drilling.

We form missingness indicators before median imputation. The corrected pilot records them from the original observed channels. A source-fitted \texttt{SimpleImputer} then supplies training-column medians to the final feature matrix, with stable zero-filled columns for entirely missing features. Global levels, imputation state and the classifier remain frozen during target inference. Each target well supplies only the whole-curve or prefix quantities admitted by its information budget. Absolute log levels and local response shape carry different physical information \citep{Asquith2004Basic,Rider2011WellLogs}; our controlled comparisons measure their representation-level effects without assigning a geological mechanism.

Local moments never pool different target wells. The feature pipeline selects the requested well's rows from the full-curve context, computes its moments and emits features only at the requested labelled depths. Source and validation wells follow the same feature definition as their corresponding targets. In contrast, the global median, global scale and imputer are learned once from training rows and reused unchanged. Quadrant ids, geographic coordinates and formation labels do not enter the classifier matrix. Measured depth enters the representation through the specified relative coordinate, and enters optional smoothing through depth order and gap segmentation.

@@FLOAT:fig:method@@
@@METHOD_TABLE@@

\subsection{Fitting and companion configurations}

We train LightGBM with at most 180 estimators, learning rate 0.05, 63 leaves, minimum child count~50, column fraction~0.8 and $L_2$ penalty~1.0. Balanced class weights address the training distribution. Early stopping monitors multiclass log-loss on the held-out validation wells with patience~30. Row bagging is disabled: \texttt{subsample}=1 and \texttt{subsample\_freq}=0. An earlier inert \texttt{subsample}=0.8 setting also lacked a positive sampling frequency; the released results follow the same no-bagging behaviour. Full-split LightGBM fits retain roughly $0.97$--$1.02\times10^6$ Protocol-1 training rows.

The same LightGBM configuration family serves the feature ablations \citep{Ke2017LightGBM}. New pilot and follow-up fits use eight threads; historical references use all available CPU threads. We do not compare runtimes across these schedules. Equal-row RF and XGBoost companions match training-row counts, while the historical RF and source-only MLP use a 120k-row ceiling. Their model capacities differ. Supplementary Methods~\ref{SI-app:impl} and~\ref{SI-app:rf} give the configurations and the covariance-only CORAL-MLP reference \citep{Sun2016CORALAAAI}; adversarial alignment provides a separate adaptation precedent \citep{Ganin2016DANN}. Scientific Python components follow \citet{Pedregosa2011sklearn}.

Optional centred majority voting uses window~7 on predictions at labelled scoring depths. A gap exceeding 1.5 times the median positive within-well depth increment breaks a segment, and votes never cross wells. The operation reads predicted classes and depth support, not target label values. We fixed window~7 before the reported sensitivity comparison; larger windows did not improve that seed-42 diagnostic (Supplementary Fig.~S5). Because the vote uses centred neighbours on scored support, we report it as an offline diagnostic. Its configuration and score rows remain distinguishable from the unsmoothed methods.

\subsection{Pooled metrics and coverage-aware tails}\label{sec:metrics}

Pooled Macro-F1 averages over classes present in the evaluated test truth. Predictions of another learned class still contribute errors, but a class absent from test truth does not add another averaging term. Open-set truth includes the never-predicted unk class. We also report a common-class measure over the intersection of train-learnable class names across repetitions, assigning absent-class F1$=0$. These two denominators answer different questions and remain separate in the released summaries (Table~\ref{tab:multirep}). Accuracy and weighted F1 are secondary under the observed class imbalance \citep{Japkowicz2002Imbalance,Sokolova2009Metrics}.

Worst-fixed is the minimum per-well Macro-F1 over the split's fixed train-learnable taxonomy, excluding unk. W10 averages the lowest $\lceil0.1n_w\rceil$ such scores; with 17 test wells it uses two wells. All wells within a split therefore share a denominator. If a closed-set well contains $k$ of the $K$ train-learnable classes, then $F1_{\mathrm{fixed},w}=(k/K)F1_{\mathrm{present},w}$. Even perfect classification of a single-facies well with $K=11$ yields about 0.091. Fixed-taxonomy tails jointly measure class coverage and prediction quality. They are not calibrated deployment-failure probabilities.

The legacy present-class minimum, Worst-true, uses each well's own observed classes. We retain it for traceability rather than substituting it for Worst-fixed. The minima select different wells in 8/18 closed Global repetitions (Supplementary Table~S9). Open-set wells also require care because the fixed taxonomy excludes unk while present-class scoring includes it. All main tail comparisons use the fixed definitions. Known-class-only open-set scores appear separately, without interpreting removal of the forced-zero unk term as unknown-class recognition.

We report means and sample standard deviations over the closed splits, paired mean changes and win counts. These summaries describe sensitivity to overlapping partitions; they are not population standard errors or hypothesis tests. The historical three-seed bootstrap intervals remain Supplementary diagnostics. We retain all prespecified follow-up seeds and all six coordinate crops, so their direction and dispersion remain visible alongside the mean. Population inference and validation on independent wells are addressed in Discussion.

We calculate pooled Macro-F1 within each test partition before averaging partition scores. We calculate the tail summaries from per-well fixed-taxonomy scores within that same partition. The well-heterogeneity analysis uses another explicit unit: it averages repeated appearances within wells and then within parents. Its equal-parent mean therefore does not reproduce pooled Macro-F1, where classwise precision and recall depend on the aggregated depth predictions. We retain these units separately throughout the results rather than interpreting a parent-averaged effect as a change in the primary pooled score.

\subsection{Matched curve sets and implementation audit}\label{sec:v2design}

The pilot uses closed seeds~42, 43 and~45 and reports open seed~44 separately. Global, corrected WellNorm, HN and Hybrid share training/validation wells and labelled test rows within each seed. The curve restriction changes only the available channels, retaining missing observations, class encoding, weights, source fitting, tree budget and validation criterion. We fit sixteen four-curve method--seed cases and four corrected sixteen-curve WellNorm cases. Twelve archived sixteen-curve Global/HN/Hybrid cases retain their original provenance; their feature matrices are unchanged by the mask correction.

The audit identified a missingness confound in r3 WellNorm: normalisation replaced missing responses with zero before legacy mask construction, leaving all-zero indicators. The correction reads the original channel. A seed-42 eight-thread legacy-mask diagnostic isolates this change from the thread setting, while an original-source reproduction gate verifies five saved Global metrics to $10^{-10}$. Together with the twenty pilot fits, these two checks account for twenty-two fit executions. Supplementary Methods~\ref{SI-sec:execution} document the counts and stored artifacts; the historical WellNorm contrasts remain explicitly mask-confounded.

We analyse well heterogeneity from saved predictions without new training. For each of the eighteen closed splits in each grouping mode, we compute HN-minus-Global present-class F1, average repeated appearances within a well, then average sibling wells within a parent. Observable descriptors include curve missing fractions, median $|\mathrm{CALI}-\mathrm{BS}|$, median $|\mathrm{DRHO}|$ and a source-standardised curve-location difference. We report Spearman associations. True class proportions and the class-conditioned mean-shift identity use target labels only after prediction (Supplementary Methods~\ref{SI-sec:heterogeneity}).

\subsection{New partitions and coordinate stresses}\label{sec:v3design}\label{sec:v3boundarydesign}

We fixed parent-grouped seeds 20261103, 20261104 and 20261105 before the follow-up fits and retain all three. Each is closed under the class-coverage rule. We fit HN and Hybrid with four and sixteen curves in each partition, giving twelve models with raw masks and the same source-validation schedule. A fitting gate reproduces five earlier four-curve Hybrid seed-42 metrics to $10^{-10}$. The primary contrast is the paired difference of differences
\[
I_s=(F1_{H,4,s}-F1_{HN,4,s})-(F1_{H,16,s}-F1_{HN,16,s}).
\]
Each term uses the same test depths within its partition. We report each $I_s$, its mean and split standard deviation. These are new partitions of FORCE, not independent datasets.

The boundary test reuses the six fitted Hybrid models. For each well, let $L=d_{\max}-d_{\min}$. We move the top, bottom or both bounds inward by 10\% or 20\% of $L$. For fractions $a,b$, retained depths lie in $[d_{\min}+aL,d_{\max}-bL]$, and the stressed coordinate is
\[
r'={d-(d_{\min}+aL)\over(1-a-b)L+10^{-6}}.
\]
Only this coordinate changes. We score baseline Hybrid, changed-coordinate Hybrid and HN on identical retained depths and class denominators, with raw curves, masks and full-curve $z$-scores fixed. The within-crop difference therefore measures the fitted learner's coordinate sensitivity.

The recovery script computes coordinates from the original float64 dataframe depths, matching feature construction, and verifies every saved baseline prediction before applying stresses. The exported FeatureBundle depths are float32 and are not used to rebuild the coordinate. No boundary scenario adds a fit or tunes a policy on test labels. Models, transformation states, predictions and matched-support counts remain in the execution records. Since full-curve normalisation moments stay fixed, this test evaluates coordinate dependence rather than the complete workflow for incomplete logs.
'''


RESULTS = r'''
\section{Results}\label{sec:exp}

\subsection{Pooled gains and well-tail trade-offs}\label{sec:primary}

Post-drilling features increased pooled mean Macro-F1 across the 18 closed well-id splits. Global-LGBM achieved $0.339\pm0.045$, HN $0.370\pm0.044$, Hybrid $0.379\pm0.037$ and Hybrid-while $0.378\pm0.041$ (Table~\ref{tab:multirep}). The configurations differ in target information as specified in Table~\ref{tab:methods}. These gains therefore describe the evaluated information settings and representations, with LightGBM held fixed. The table retains common-class scores and both fixed-tail summaries alongside pooled Macro-F1.

Higher pooled performance did not consistently improve the worst well. Hybrid exceeded HN in mean Macro-F1, but its Worst-fixed was 0.101 versus 0.105 for HN (Fig.~\ref{fig:pareto}). HN improved both pooled Macro-F1 and Worst-fixed relative to Global, with paired mean Macro-F1 gain $+0.032$ and wins in 15/18 closed splits (Table~\ref{tab:pairedmr}). The joint view exposes this distinction directly: the highest pooled mean and the strongest minimum belong to different representations in the well-id comparison.

Parent grouping changed the tail ordering. Global's grouped Macro-F1, $0.340\pm0.047$, stayed close to its well-id result, while the mean Hybrid-minus-HN Worst-fixed difference changed from $-0.004$ to $+0.006$, with 13/18 positive grouped contrasts (Tables~\ref{tab:multirepparent}--\ref{tab:pairedmr}). This is a grouping-sensitivity result on overlapping partitions. The fixed taxonomy still gives every well in a split the same class denominator, and a change in its minimum does not by itself establish improved geological interpretation for each target.

The main ordering persisted in the retained sensitivities. Excluding ablation seed~42 gave Macro-F1 0.335/0.368/0.377 for Global/HN/Hybrid over 17 closed splits; including all 20 splits gave 0.337/0.360/0.370 (Supplementary Table~S2). Neither sensitivity replaces the prespecified closed summary. Open seeds retain their unseen-class penalty, while the historical three-seed tables supply ablation detail. The parent Pareto panel and complete per-seed companion results appear in Supplementary Figs.~\ref{SI-fig:paretoparent} and Tables~\ref{SI-tab:multiseed}--\ref{SI-tab:whilems}.

@@FLOAT:tab:multirep@@
@@FLOAT:tab:multirepparent@@
@@FLOAT:tab:pairedmr@@
@@FLOAT:fig:pareto@@

\subsection{The depth-split contrast combines several changes}\label{sec:embargo}\label{sec:parent}

The naive depth-point baseline scored 0.737, compared with the primary well-held-out mean 0.339 (Table~\ref{tab:multirep}; Supplementary Table~\ref{SI-tab:matched}). Within the seed-42 diagnostics, contiguous depth-block hold-out scored 0.497, a change of $-0.240$ without changing well identity. The test depths and class proportions also changed. The matched target-row diagnostic instead scored 0.436 before adding target non-test labels and 0.464 after adding them. Equal-row substitution scored 0.459 on the published draw; ten subsample seeds gave mean gain $+0.020\pm0.006$.

Local separation produced a smaller change in the separate fixed-test-block embargo study. Mean Macro-F1 changed from $0.425\pm0.006$ at 0\,m to $0.410\pm0.004$ at 20\,m (Supplementary Table~S6). Removing the related \texttt{34/5-1 S} wellbore from seed-44 training changed Macro-F1 from 0.326 to 0.324 and legacy Worst-true from 0.082 to 0.093 (Supplementary Table~S7). These diagnostics identify which design elements changed and how their scores responded. They do not isolate a pure neighbour-leakage penalty or partition the headline gap additively.

\subsection{Restricting curves changes the representation benefit}\label{sec:v2pilot}

Four-curve inputs reduced mean Macro-F1 for every method in the matched pilot. Across closed seeds~42/43/45, Global scored $0.296\pm0.031$, corrected WellNorm $0.277\pm0.044$, HN $0.321\pm0.049$ and Hybrid $0.364\pm0.044$ (Table~\ref{tab:v2core}). Their sixteen-curve means were 0.371, 0.320, 0.401 and 0.409. The paired four-minus-sixteen changes were $-0.075$, $-0.042$, $-0.080$ and $-0.045$, respectively. Since the labelled cohort stayed fixed, the comparison describes sensitivity to the input curves rather than to dropping incomplete rows.

Relative depth offset part of this reduction for Hybrid. Adding it to HN increased mean Macro-F1 by $+0.043$ with four curves and $+0.008$ with sixteen, giving an initial difference of gains of $+0.035$ (Fig.~\ref{fig:v2core}). The pooled and Worst-fixed changes differed under the same row support. Open seed~44 remained a separate stress, with unknown classes retained and never predicted (Supplementary Table~S11). The initial pattern motivated the new-partition check below.

The mask audit changed the interpretation of historical WellNorm comparisons. On seed~42, restoring raw indicators increased sixteen-curve WellNorm Macro-F1 from 0.293 to 0.356. The eight-thread legacy-mask diagnostic reproduced archived Macro-F1 and Worst-fixed to $10^{-10}$, isolating the mask change on that seed (Supplementary Table~S12). Across the three closed pilot splits, new-minus-archived Macro-F1 averaged $+0.014$, ranged from $-0.013$ to $+0.063$, and was positive in one split. Restoring information made the feature comparison consistent; it did not produce a uniform accuracy gain. We use this corrected pilot, rather than the mask-confounded historical gaps, to interpret WellNorm.

The matching also clarifies what the curve comparison controls. Every method sees the same source/test assignment and scored depths within a seed, and retains the same missingness definition. The sixteen-curve Global/HN/Hybrid references reuse the archived fitted models; their feature-equivalence checks protect that reuse. Corrected WellNorm and all four-curve cases are new fits from the recorded pilot phase. We distinguish those provenances in the table rather than presenting reused references as additional independent experiments. The observed difference of relative-depth gains is a paired contrast within these settings.

@@FLOAT:tab:v2core@@
@@FLOAT:fig:v2core@@

\subsection{New partitions narrow the curve-set contrast}\label{sec:v3replication}

The three new parent-grouped partitions yielded a smaller average difference of relative-depth gains. Hybrid-minus-HN Macro-F1 averaged $+0.019$ with four curves and $+0.008$ with sixteen. The paired difference of differences averaged $+0.011$, ranged from $-0.018$ to $+0.045$, and was positive in 2/3 partitions (Fig.~\ref{fig:v3new}; Supplementary Table~S14). The new scores therefore retained substantial split-to-split variation rather than reproducing one curve-count ordering in every partition.

Table~\ref{tab:v3new} gives the baseline means for these twelve newly fitted models. Hybrid scored $0.402\pm0.035$ with four curves and $0.415\pm0.038$ with sixteen; HN scored $0.383\pm0.041$ and $0.406\pm0.057$. The common-class set is computed within this follow-up and need not equal the initial pilot's set. Figure~\ref{fig:v3new} keeps the phases separate, showing both the original $+0.035$ contrast and the smaller new mean. Grouping and partition both change between phases, so the reduction cannot be assigned to parent grouping alone.

The paired design answers a narrower question than comparing the two Hybrid means alone. Within a curve set, HN and Hybrid share the underlying curves, masks and well-normalised features; adding relative depth defines their contrast. Comparing those two gains across curve sets then asks whether the extra coordinate changes performance differently when fewer channels are available. Each partition contributes one paired value. The negative value in one of the three new partitions remains part of the result and prevents the average from being read as a uniform restricted-curve advantage.

@@FLOAT:tab:v3new@@
@@FLOAT:fig:v3new@@

\subsection{The fitted learner is sensitive to interval coordinates}\label{sec:v3boundary}

Changing relative depth reduced four-curve mean Macro-F1 in every prespecified crop, by 0.011--0.052 (Fig.~\ref{fig:v3boundary}). Sixteen-curve mean changes ranged from $-0.024$ to $-0.001$. Across model--crop cases, 30/36 changes were negative. Retained rows ranged from 59.1\% to 98.9\% across split--crop comparisons (Supplementary Tables~S15--S16). Each difference compares changed and baseline predictions on the same retained rows; changes in the crop's class composition cannot explain that within-crop difference.

Two coordinate stresses reversed the mean four-curve advantage over HN. Under a 20\% bottom crop, the original matched-support advantage was $+0.035$, becoming $-0.017$ after recalculating the coordinate. Under a two-sided 20\% crop, it changed from $+0.034$ to $-0.018$ (Supplementary Table~S15). The retained physical samples, raw measurements, masks and normalisation moments were identical within each comparison. The result attributes the prediction change to the coordinate supplied to the fitted learner.

This controlled test establishes interval-coordinate dependence within the evaluated representation. It does not measure the performance of a complete partial-log pipeline, because the $z$-scores still use full curves. The distinction matters operationally: changing the coordinate and recomputing every target-dependent feature are different interventions. We report the former here, preserve all six crops, and use the latter to define a validation requirement in Discussion.

Two comparisons must also remain distinct within a crop. The stress effect compares changed-coordinate Hybrid with baseline Hybrid on the retained rows. The HN comparison asks whether either Hybrid coordinate improves on a model that omits relative depth. Restricting the rows first gives all three predictions the same physical support and class denominator. It removes crop-composition differences from the within-crop contrast, but it does not remove them when comparing absolute scores between different crops. The Supplementary tables therefore retain row, well and class counts for every scenario.

@@FLOAT:fig:v3boundary@@

\subsection{Benefits vary across wells and facies}\label{sec:v2heterogeneity}\label{sec:rare}

The average HN gain concealed losses on individual parent-averaged wells. After averaging repeated test appearances and sibling wells, HN exceeded Global in present-class F1 for 75/107 parents under well-id splitting. The mean change was $+0.027$, with median $+0.024$ and 10th--90th percentiles $-0.043$ to $+0.100$. The parent-grouped companion had 78/106 positive effects, mean $+0.027$ and median $+0.017$ (Supplementary Fig.~\ref{SI-fig:v2heterogeneity}). Equal parent weighting makes these quantities distinct from the pooled split scores.

Observable descriptors had weak rank associations with the paired effect. Under well-id splitting, correlations were $-0.150$ for four-curve location difference, $+0.109$ for four-curve missingness, $-0.078$ for sixteen-curve missingness, $+0.034$ for median $|\mathrm{CALI}-\mathrm{BS}|$ and $+0.178$ for median $|\mathrm{DRHO}|$. Available parent counts varied by descriptor. The class-conditioned shift analysis also remained descriptive: it uses true target labels and algebraically separates composition and within-class terms, rather than identifying calibration or geological causes (Supplementary Fig.~S10). Neither analysis evaluates a prospective method-selection rule.

Rare-class errors persisted in the seed-42 diagnostic. Hybrid increased F1 for Sandstone and Sandstone/Shale, while Chalk and Dolomite remained near zero (Fig.~\ref{fig:perclass}a). The confusion matrix assigned Chalk mainly to Limestone, Dolomite to Shale/Sandstone, and about 62--65\% of true Tuff depths to Shale (Fig.~\ref{fig:rarecm}b). These observed errors support interpreter review of rare-facies predictions. Candidate run-length and disagreement flags remain uncalibrated proposals, with their definitions retained in Supplementary Methods~\ref{SI-sec:reviewflags}.

@@RARE_FIGURE@@

\subsection{Causal prefixes and companion models}\label{sec:while}\label{sec:multiseed}

Hybrid-while achieved primary mean Macro-F1 $0.378\pm0.041$ (Table~\ref{tab:multirep}). The prefix support was already substantial: on seed~42, $220733/220839$ scored depths had at least 100 preceding/current full-curve samples (Supplementary Table~S1). The historical three-seed mean was 0.412 versus 0.371 for Global, with Worst-fixed 0.111 versus 0.104 (Supplementary Table~\ref{SI-tab:whilems}). This result concerns causal normalisation of completed wireline curves under the observed coverage. Sparse-prefix and early-bit stresses remain separate Supplementary diagnostics.

The inductive companions further delimit the role of the fixed backbone. Equal-row RF-full achieved historical three-seed mean Macro-F1 0.402, XGBoost 0.377 and Global-LGBM 0.371 (Supplementary Table~\ref{SI-tab:multiseed}). We therefore use LightGBM for controlled representation comparisons, without claiming that it wins an algorithm benchmark. Historical CPU timing records show a median 211\,s per fit across the 18 closed splits and six LightGBM methods, with IQR 174--256\,s; the record includes archived WellNorm. Hardware and per-model timings appear in Supplementary Methods~\ref{SI-sec:execution}. The eight-thread follow-up is not included in that runtime comparison.
'''

DISCUSSION = r'''
\section{Discussion}\label{sec:discussion}\label{sec:forcecmp}

\subsection{What a cross-well score establishes}

The results connect performance to information access, input support and interval definition. The primary pooled gain appears across the evaluated post-drilling representations, while the well-tail ordering changes with the representation and grouping unit (Tables~\ref{tab:multirep}--\ref{tab:pairedmr}). The restricted-curve comparison and coordinate stresses add a more specific finding: relative depth has a partition-dependent benefit and an interval-dependent coordinate effect (Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}). Reporting these conditions makes the score interpretable beyond a single pooled mean.

The distinction complements existing lithology benchmarks. \citet{Carvalho2026LithologyBenchmark} standardise four-log processing and well-level model comparison; regional UDA and DG studies distinguish transfer assumptions \citep{Chang2021GRSL,Xu2022JNGSE,Sun2023Energies,Li2024TAI}. Our experiments hold a classifier family fixed and expose what changes when evaluation, target statistics and curve availability change. The depth-split contrast measures combined protocol optimism. The coordinate test instead changes one feature on matched support, giving a controlled explanation of the fitted model's response. These designs answer different questions and should retain their own denominators and conditions.

The input and coordinate results also distinguish a statistical benefit from a geological explanation. Relative depth specifies position inside the observed interval; it does not align formation boundaries between wells. Likewise, a well-wise response shift combines the sampled lithologies with within-class measurement differences. The class-conditioned identity describes that mixture using target labels, but does not identify its physical causes (Supplementary Fig.~S10). The present evidence supports reporting and testing these feature dependencies. Attribution to tool calibration, fluids or depositional structure requires measurements that separate those mechanisms.

The raw-mask correction reinforces the need to compare information as well as nominal feature names. Historical WellNorm lost missingness information, while Global and Hybrid retained it. A nominal Hybrid--WellNorm contrast therefore changed more than global levels. The corrected pilot restores comparable indicators and yields gains of both signs across its three closed splits (Table~\ref{tab:v2core}; Supplementary Table~S12). This audit changes attribution, even where it does not improve accuracy. Keeping the corrected and archival results separate prevents an implementation discrepancy from becoming a claim about normalisation itself.

\subsection{Implications for reporting and interpretation}\label{sec:ops}

A cross-well report should state the well split, class-coverage rule, feature information budget and training count before presenting scores. The filled reporting record pairs Macro-F1 with Worst-fixed and W10 and retains the parent-grouped companion (Supplementary Tables~\ref{SI-tab:checklist}--\ref{SI-tab:accept}). Its role is reproducible disclosure. Because fixed tails combine class coverage with prediction quality, low scores identify wells requiring closer inspection without assigning a calibrated failure probability.

Input access determines which construction is admissible. Global-LGBM applies when target aggregates are unavailable. Full-curve HN and Hybrid require post-drilling observations, including the interval bounds for relative depth. Hybrid-while uses completed-wireline prefixes under the coverage actually observed here (Table~\ref{tab:methods}; Supplementary Table~S1). These distinctions belong in the reporting record alongside any interval crop or curve restriction. The rare-class errors and heterogeneous well effects support an interpretation-assistance role, with geological control and human review retained (Fig.~\ref{fig:perclass}; Supplementary Fig.~\ref{SI-fig:v2heterogeneity}).

\subsection{Limitations and future work}\label{sec:limitations}

All primary and follow-up partitions reuse FORCE wells. Their standard deviations and win counts describe split sensitivity, not population uncertainty. Independent-corpus validation and nested well-level or hierarchical resampling are required for population estimates. The locked schedule predates final provenance capture but was not formally preregistered. The initial pilot has three closed splits and one open stress, and the follow-up adds three parent-grouped partitions. Since grouping and partition both change between phases, a controlled grouping study must vary those factors separately. Full geographic and leave-one-well-out schedules would also extend the two-quadrant and six-well stresses.

The boundary test changes only relative depth and retains full-curve normalisation. A full incomplete-log evaluation requires truncating the actual observations, recomputing all target-dependent features and scoring partial-log predictions. Likewise, the causal-prefix result requires validation on sparse prefixes and channels available during real MWD/LWD before supporting a while-drilling deployment claim. The present code and recorded supports make those differences explicit.

Corrected WellNorm evidence covers the matched pilot, leaving the historical eighteen-split mean mask-confounded. A corrected full-schedule rerun is required to replace that archive. Equal-row RF/XGBoost comparisons match rows rather than model capacity; strong UDA, source reweighting and confidence weighting require controlled comparisons of their own. The covariance-only CORAL reference also uses a different row budget and does not establish a general failure of adaptation.

Finally, the classifier has no unknown-class detector or calibrated rejection on open splits. The candidate review flags have no measured sensitivity, specificity or selective-prediction guarantee. Validation on separate wells must assess these properties before model agreement allocates review effort. The descriptor and label-dependent shift analyses supply descriptive context, not a prospective selector. Developing such a rule requires separate development and evaluation wells. These next tests address the uncertainties that remain after the current protocol, feature and coordinate comparisons.

\section{Conclusions}\label{sec:conc}

We demonstrate that explicit information budgets and well-level assessment change the interpretation of FORCE lithofacies scores. Global-LGBM achieves Macro-F1 $0.339\pm0.045$ over 18 closed well-held-out repetitions, compared with 0.737 under naive depth splitting; post-drilling Hybrid variants achieve 0.370--0.379 (Table~\ref{tab:multirep}). Their pooled gains do not produce one consistent well-tail ordering (Table~\ref{tab:pairedmr}).

Relative-depth gains depend on the tested inputs and interval coordinate. The four-minus-sixteen-curve difference averages $+0.011$ across three new parent-grouped partitions, with two of three positive. Matched-support coordinate changes reverse the mean four-curve HN advantage under bottom and two-sided 20\% crops (Figs.~\ref{fig:v3new}--\ref{fig:v3boundary}). The resulting framework links scores to the curves, statistics, interval bounds and class coverage used to obtain them, with saved predictions and execution records supporting reproducibility.
% CORE_TEXT_END
'''

AVAILABILITY = r'''
\section*{Data availability}
The public FORCE~2020 release supplies the well logs and labels \citep{Bormann2020FORCE} (\href{https://doi.org/10.5281/zenodo.4351156}{10.5281/zenodo.4351156}). The separate \texttt{GSE\_review\_materials\_20261103.zip} contains processed input tables, split definitions, results, saved follow-up models and predictions, with configuration/data/source hashes. Its inventory distinguishes historical r3 results from the corrected pilot and new-partition follow-up. The current manuscript package supplies a filtered primary reporting record and records the legacy WellNorm qualification explicitly. Independent-corpus validation remains a requirement in Section~\ref{sec:limitations}.

\section*{Code availability}
The review archive includes CPU Python source, locked dependencies, tests, reproduction instructions, feature-equivalence checks and execution-source snapshots. New experiments used Python~3.11.5 and the recorded dependency versions. Historical r3 and v2/v3 results retain separate provenance. The current compact package adds revised text, an editable vector overview and audit records; it does not rerun the reported experiments. The earlier \url{https://github.com/yangy0907/cross_well_dg} is a historical locator. The archive has a SHA-256 sidecar; no software DOI or public deposition is asserted.
'''


def expand(text):
    for name in ['v2_methods', 'v2_results', 'v3_methods', 'v3_results']:
        text = text.replace(r'\input{' + name + '}', (SRC / f'{name}.tex').read_text(encoding='utf-8'))
    return text


def main():
    DST.mkdir(exist_ok=True)
    for path in SRC.iterdir():
        if path.suffix in {'.bib', '.cls', '.sty', '.bst'} or path.name == 'highlights.txt':
            shutil.copy2(path, DST / path.name)
    for path in (SRC / 'figs').iterdir():
        if path.is_file() and not path.name.startswith('figM_method_overview'):
            shutil.copy2(path, DST / 'figs' / path.name)
    for name in ['table_v2_common_core.tex', 'table_v3_new_splits.tex']:
        shutil.copy2(SRC / name, DST / name)
    old = expand((SRC / 'main.tex').read_text(encoding='utf-8'))
    floats = {}
    for kind in ['table', 'figure']:
        for match in re.finditer(r'\\begin\{' + kind + r'\}.*?\\end\{' + kind + r'\}', old, re.S):
            label = re.search(r'\\label\{([^}]+)\}', match[0])
            if label:
                floats[label[1]] = match[0]
    finish(old, floats)
    update_references()


def finish(old, floats):
    # All numerical blocks survive either in the main manuscript or SI.
    retained = {'tab:multirep', 'tab:multirepparent', 'tab:pairedmr', 'tab:v2core', 'tab:v3new',
                'fig:method', 'fig:pareto', 'fig:v2core', 'fig:v3new', 'fig:v3boundary'}
    captions = {
        'fig:method': r'Information flow for the main LightGBM comparisons. Source training rows fit the global transforms and classifier; separate validation wells provide early stopping. Target facies enter evaluation and class-coverage bookkeeping, not fitting or tuning. The three feature budgets are parallel alternatives. Curve-availability comparisons require fitting their specified models; interval-coordinate stresses reuse fitted models and freeze all other features. Full-curve moments remain fixed in those stresses.',
        'tab:multirep': r'Primary well-id results: mean$\pm$sample SD across 18 overlapping closed splits. $C_{\mathrm{common}}$ uses 11 shared train-learnable facies with absent-class F1$=0$. Worst-fixed and W10 use the within-split fixed taxonomy. SD describes split sensitivity. Hybrid+Smooth is an offline score-support diagnostic.',
        'tab:multirepparent': r'Parent-grouped companion: 18 overlapping closed splits, with metrics defined as in Table~\ref{tab:multirep}. Open seeds~44 and~127 remain separate stresses.',
        'tab:pairedmr': r'Paired closed-split changes, mean$\pm$sample SD and positive counts over 18 overlapping repetitions. These are descriptive split-sensitivity summaries. HN denotes Hybrid-norel. Fixed tails include class coverage.',
        'fig:pareto': r'Pooled Macro-F1 and Worst-fixed for the four main information-budget configurations. Error bars show sample SD across 18 overlapping closed well-id splits, not population confidence intervals. The historical mask-confounded WellNorm point and optional smoothing diagnostic are excluded from this main comparison.',
        'tab:v2core': r'Matched-row pilot: means$\pm$sample SD on closed seeds~42/43/45. Raw missingness is retained. Sixteen-curve Global/HN/Hybrid rows reuse archived references; all four-curve and corrected sixteen-curve WellNorm rows are newly fitted. The common-class set is defined within this pilot.',
        'fig:v2core': r'Curve restriction under matched labelled rows in the pilot. Error bars describe split SD. Worst-fixed jointly measures class coverage and prediction quality.',
        'tab:v3new': r'Twelve new fits on parent-grouped seeds 20261103/20261104/20261105. Values are means$\pm$sample SD over three overlapping closed partitions. The common-class set is computed within this follow-up.',
        'fig:v3new': r'Relative-depth benefit with four and sixteen curves. Lines join effects within the same split. The original pilot and new parent-grouped partitions reuse FORCE and are shown as separate phases.',
        'fig:v3boundary': r'Inference-only relative-depth coordinate stresses. Lines are new partitions; diamonds are their means. Each changed/baseline comparison uses identical retained depths. Other features and full-curve normalisation moments remain fixed. Top/bottom/both move the indicated bounds by 10\% or 20\% of the original span.',
    }
    # Balanced braces are required because captions contain LaTeX commands.
    def caption(block, replacement):
        start = block.index(r'\caption{')
        p, level = start + len(r'\caption{'), 1
        end = p
        while level:
            level += (block[end] == '{') - (block[end] == '}')
            end += 1
        return block[:p] + replacement + block[end-1:]
    core = INTRO + METHODS + RESULTS + DISCUSSION
    for key in retained:
        block = caption(floats[key], captions[key])
        if key == 'fig:pareto':
            block = block.replace('fig_followup_pareto_well_id.pdf', 'fig_compact_pareto.pdf')
        core = core.replace('@@FLOAT:' + key + '@@', block)
    method_table = r'''\begin{table}[width=.98\linewidth,pos=htbp]
\caption{Main representations and target information. $G$: source-standardised levels; $Z$: whole-well $z$; $Z_{\mathrm{prefix}}$: causal expanding $z$; $r$: relative depth; $M$: missingness indicators. Target facies do not enter fitting. Corrected WellNorm is evaluated in the matched pilot.}
\label{tab:methods}
\begin{tabular*}{\tblwidth}{@{}lll@{}}
\toprule
Model & Features & Target support \\
\midrule
Global-LGBM & $G,M$ & No target aggregates \\
Hybrid-norel (HN) & $G,Z,M$ & Full observed curves \\
Hybrid & $G,Z,r,M$ & Full observed interval \\
Hybrid-while & $G,Z_{\mathrm{prefix}},M$ & Past and current depths \\
Corrected WellNorm & $Z,r,M$ & Full observed interval; pilot \\
\bottomrule
\end{tabular*}
\end{table}'''
    rare = r'''\begin{figure}[width=.95\linewidth,pos=htbp]
\centering
\makebox[\linewidth][l]{\small\textbf{(a)}}
\includegraphics[width=.88\linewidth]{figs/fig_followup_per_class_f1.png}
\par\smallskip
\makebox[\linewidth][l]{\small\textbf{(b)}}
\includegraphics[width=.88\linewidth]{figs/fig_followup_rare_confusion.png}
\caption{Seed-42 class diagnostics. (a) Global-LGBM and Hybrid per-class F1, with support beneath the names. (b) Hybrid row-normalised rare-class confusion. Chalk and Dolomite remain near zero F1; Tuff is frequently assigned to Shale. The panels retain the original plotted data.}
\label{fig:perclass}\label{fig:rarecm}
\end{figure}'''
    core = core.replace('@@METHOD_TABLE@@', method_table).replace('@@RARE_FIGURE@@', rare)
    core = core.replace('0.335/0.368/0.377', '0.335, 0.368 and 0.377').replace('0.337/0.360/0.370', '0.337, 0.360 and 0.370')
    core = core.replace(r'Tables~\ref{SI-tab:multiseed}--\ref{SI-tab:whilems}', r'Tables~\ref{SI-tab:multiseed} and~\ref{SI-tab:whilems}').replace(r'Supplementary Figs.~\ref{SI-fig:paretoparent}', r'Supplementary Fig.~\ref{SI-fig:paretoparent}')
    assert '@@' not in core
    preamble = old[:old.index(r'\begin{abstract}')]
    preamble = preamble.replace(r'\usepackage{hyperref}', r'\usepackage{xr-hyper}' + '\n' + r'\usepackage{hyperref}' + '\n' + r'\usepackage{placeins}' + '\n' + r'\externaldocument[SI-]{supplementary}')
    abstract = re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', old, re.S)[0]
    front = old[old.index(r'\end{abstract}')+len(r'\end{abstract}'):old.index(r'\section{Introduction}')]
    declarations = old[old.index(r'\section*{Declaration of competing interests}'):]
    main_text = preamble + abstract + front + core + AVAILABILITY + declarations
    # Ensure Results floats are placed before Discussion, rather than interrupting it.
    main_text = main_text.replace(r'\section{Discussion}', r'\FloatBarrier' + '\n' + r'\section{Discussion}')
    (DST / 'main.tex').write_text(main_text, encoding='utf-8')

    si = (SRC / 'supplementary.tex').read_text(encoding='utf-8')
    si = si[:si.index(r'\end{document}')]
    si = si.replace(r'\usepackage{hyperref}', r'\usepackage{xr-hyper}' + '\n' + r'\usepackage{hyperref}' + '\n' + r'\externaldocument[M-]{main}')
    si += r'''
\clearpage
\renewcommand{\thesection}{S\arabic{section}}
\setcounter{section}{0}
\section{Additional methods and execution records}\label{sec:execution}

The original-source Global reproduction gate, twenty new curve-pilot fits and one same-thread mask diagnostic account for twenty-two executions in the pilot phase. Twelve sixteen-curve comparator rows are reused references. The follow-up adds twelve fits and one successful reproduction gate; one earlier gate attempt failed during native Unicode-path model persistence. Python model-text writing and readback resolved saving without changing the schedule. Thus the follow-up has thirteen successful fits and one failed persistence attempt. Boundary inference reuses saved models and adds no fits. The execution manifests retain these distinctions.

Historical CPU timings use Windows workstations, typically 12 cores, 16 threads and at least 32 GB RAM, with LightGBM/sklearn using all available threads. Median runtime across the eighteen closed well-id splits and six LightGBM methods is 211 s (IQR 174--256): Global 181 s (143--199), Hybrid 236 s (179--253), HN 239 s (184--260). The CSV seconds fields supply the individual records. New fits use eight threads and are not compared to this runtime distribution.

\subsection{Well descriptors and post-hoc analysis}\label{sec:heterogeneity}
'''
    v2m = (SRC / 'v2_methods.tex').read_text(encoding='utf-8')
    si += v2m[v2m.index('Saved predictions from'):v2m.index('Including the earlier original-source')]
    si += r'''
\subsection{Candidate interpreter-review flags}\label{sec:reviewflags}
The proposed flags are runs of at least three predicted Chalk, Dolomite or Tuff depths and Global--Hybrid disagreements spanning at least five depths. The rare-class diagnostics motivate these candidates, but their sensitivity, specificity and selective-prediction properties have not been evaluated. Agreement between methods does not establish geological reliability. Core observations, independent geological control and human interpretation remain the relevant validation sources.

\subsection{Coordinate precision and persistence checks}
Coordinates are rebuilt from original float64 dataframe depths, rather than exported float32 FeatureBundle depths. The recovery script verifies saved-model baseline predictions on every test row before applying crops. Only relative depth changes; full-curve moments and every other feature remain fixed. The recorded support check compares all three methods on identical retained rows within each crop.
'''
    # Preserve the original detailed implementation sections and their numeric RF table.
    appendix = old[old.index(r'\appendix')+len(r'\appendix'):old.index(r'\section*{Data availability}')]
    rf = floats['tab:rf']
    appendix = appendix.replace(rf, '')
    appendix = appendix.replace('cross-algorithm ranking remains abandoned. Equal-row strong UDA remains deferred.',
                                'these configurations document the specified row budgets. Strong UDA requires a separate controlled comparison.')
    appendix = appendix.replace('The stronger 256/128-hidden-unit variant uses up to 200k training rows and 80k target rows, yet remains below LightGBM.',
                                'A separate 256/128-hidden-unit variant uses up to 200k training rows and 80k target rows; it also has an unequal budget.')
    si += appendix
    si += '\n\\FloatBarrier\n\\section{Supporting tables and figures}\n'
    table_order = ['tab:methods', 'tab:coverage', 'tab:regime', 'tab:matched', 'tab:p1',
                   'tab:whilems', 'tab:multiseed', 'tab:checklist', 'tab:p2', 'tab:p3',
                   'tab:forcecmp', 'tab:accept', 'tab:rf']
    for key in table_order:
        block = floats[key]
        if key == 'tab:methods':
            block = block.replace(r'[width=.95\linewidth,pos=htbp]', r'[width=.99\linewidth,pos=htbp]')
            block = block.replace(r'\label{tab:methods}', r'\label{tab:methodsfull}')
            block = block.replace(r'\label{tab:methodsfull}', r'\label{tab:methodsfull}' + '\n' + r'\small')
            block = caption(block, r'Full historical and companion configuration catalogue. The nominal WellNorm/Hybrid feature grid is mask-confounded in legacy r3 WellNorm branches; the corrected pilot reads raw masks. Main-text Table~\ref{tab:methods} gives the core representations. Row budgets and alignment support differ for the companion algorithms.')
        si += '\n' + block + '\n'
    for key in ['fig:paretoparent', 'fig:class', 'fig:leak', 'fig:p1', 'fig:multiseed', 'fig:v2heterogeneity']:
        si += '\n' + floats[key] + '\n'
    # Make cross-document references explicit without colliding with SI-local labels.
    si_labels = set(re.findall(r'\\label\{([^}]+)\}', si))
    main_labels = set(re.findall(r'\\label\{([^}]+)\}', main_text))
    def si_ref(match):
        key = match[1]
        if key in si_labels:
            return match[0]
        assert key in main_labels, ('Unknown SI reference', key)
        return r'\ref{M-' + key + '}'
    si = re.sub(r'\\ref\{([^}]+)\}', si_ref, si)
    si += '\n\\FloatBarrier\n\\bibliographystyle{unsrtnat}\n\\bibliography{refs}\n\\end{document}\n'
    (DST / 'supplementary.tex').write_text(si, encoding='utf-8')
    (DST / 'audit/relocation_map.json').write_text(json.dumps({
        'retained_core_floats': sorted(retained),
        'moved_tables': table_order,
        'moved_figures': ['fig:paretoparent', 'fig:class', 'fig:leak', 'fig:p1', 'fig:multiseed', 'fig:v2heterogeneity'],
        'combined_panels': ['fig:perclass', 'fig:rarecm'],
        'new_core_table': 'tab:methods',
        'implementation_appendix': 'Supplementary sections',
    }, indent=2), encoding='utf-8')
    from revise_evidence_style import words
    before = old[old.index(r'\begin{abstract}'):old.index(r'\appendix')]
    after = main_text[main_text.index(r'\begin{abstract}'):main_text.index('% CORE_TEXT_END')]
    counts = {'before_core_tokens': words(before), 'after_core_tokens': words(after),
              'reduction_percent': round(100*(1-words(after)/words(before)), 2),
              'scope': 'Abstract through Conclusions, excluding tables/figures, math and reference fields. Both versions use the same English-token procedure. Appendix/availability/declarations/references are excluded.',
              'main_figures': main_text.count(r'\begin{figure}'), 'main_tables': main_text.count(r'\begin{table}')}
    (DST / 'audit/compression.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    # Match every original tabular body against the combined package, including input files.
    all_new = main_text + si
    for name in ['table_v2_common_core.tex', 'table_v3_new_splits.tex']:
        all_new += (DST / name).read_text(encoding='utf-8')
    bodies = re.findall(r'\\begin\{(tabular\*?)\}(.*?)\\end\{\1\}', old, re.S)
    for kind, body in bodies:
        assert body in all_new, ('Lost numerical table', body[:100])
    for path in DST.glob('*.tex'):
        raw = path.read_bytes()
        assert not re.search(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]|\r(?!\n)', raw), ('Control character', path.name)
    print(json.dumps(counts, indent=2))


def update_references():
    from audit_compact_references import parse_bib
    entries = parse_bib((SRC / 'refs.bib').read_text(encoding='utf-8'))
    updates = {
        'Bormann2020FORCE': {'title': '{FORCE} 2020 Well well log and lithofacies dataset for machine learning competition'},
        'Carvalho2026LithologyBenchmark': {'author': r'Carvalho, Henrique M. O. and Daniel, Henrique and Korenchendler, Andr{\'e} and Sobreira, Matheus C. A. and Machado, Alexei M. C.'},
        'Friedman2001GBM': {'doi': '10.1214/aos/1013203451'},
        'Ke2017LightGBM': {'url': 'https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html'},
        'Pedregosa2011sklearn': {'number': '85', 'url': 'https://www.jmlr.org/papers/v12/pedregosa11a.html'},
        'Ganin2016DANN': {'url': 'https://jmlr.org/papers/v17/15-239.html'},
        'Sagawa2020GroupDRO': {'url': 'https://iclr.cc/virtual_2020/poster_ryxGuJrFvS.html'},
        'Koh2021WILDS': {'url': 'https://proceedings.mlr.press/v139/koh21a.html'},
        'Gulrajani2021DomainBed': {'url': 'https://openreview.net/forum?id=lQdXeXDoWtI'},
        'Asquith2004Basic': {'url': 'https://store-assets.aapg.org/documents/previews/707Mth16/CHAPTER01.PDF'},
        'Rider2011WellLogs': {'author': 'Rider, Malcolm H. and Kennedy, Martin', 'address': 'Glasgow',
                              'isbn': '9780954190682', 'url': 'https://catalogue.bnf.fr/ark:/12148/cb42702380r'},
        'Avseth2010QSI': {'year': '2005', 'doi': '10.1017/CBO9780511600074', 'note': 'First print publication; online edition 2010'},
    }
    changes = []
    for key, fields in updates.items():
        for field, value in fields.items():
            previous = entries[key].get(field)
            entries[key][field] = value
            changes.append({'key': key, 'field': field, 'before': previous, 'after': value})
    bib = ['%% Bibliography audited 2026-11-04 against DOI registries and primary records.',
           '%% See audit/reference_verification.md for sources, scope and version notes.', '']
    for key, fields in entries.items():
        bib.append('@' + fields['entry_type'] + '{' + key + ',')
        bib += ['  ' + field + ' = {' + value + '},' for field, value in fields.items() if field != 'entry_type']
        bib += ['}', '']
    for name in ['refs.bib', 'supplementary-refs.bib']:
        (DST / name).write_text('\n'.join(bib), encoding='utf-8')
    (DST / 'audit/bibliography_changes.json').write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding='utf-8')
    highlights = [
        'Well-held-out scores depend on admissible target-well information.',
        'Relative-depth gains vary across log sets and new parent-grouped splits.',
        'Boundary-coordinate stresses reverse mean gains in two tested crops.',
        'Fixed-taxonomy tails combine class coverage with predictive performance.',
    ]
    (DST / 'highlights.txt').write_text('\n'.join(highlights) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
