"""Assemble a separate v3 manuscript after the frozen numerical extension finishes."""
from pathlib import Path
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/gse_v2'
DST = ROOT / 'paper/gse_v3'
OUT = ROOT / 'results/v3/relative_depth_validation_run2'


def table(headers, rows, columns):
    return '\n'.join([r'\begin{tabular}{@{}' + columns + r'@{}}', r'\toprule',
                       ' & '.join(headers) + r' \\', r'\midrule'] +
                       [' & '.join(str(v) for v in row) + r' \\' for row in rows] +
                       [r'\bottomrule', r'\end{tabular}'])


def main():
    notes = json.loads((OUT / 'summary_notes.json').read_text(encoding='utf-8'))
    inter = pd.read_csv(OUT / 'new_split_interactions.csv')
    base = pd.read_csv(OUT / 'baseline_metrics.csv')
    stress = pd.read_csv(OUT / 'boundary_metrics.csv')
    bs = pd.read_csv(OUT / 'boundary_summary.csv')
    summary = pd.read_csv(OUT / 'closed_summary.csv')
    if notes['n_closed_splits'] != 3:
        raise RuntimeError('Revise prose for actual regime counts rather than silently substituting seeds')
    DST.mkdir(parents=True, exist_ok=True)
    names = ['main.tex', 'supplementary.tex', 'v2_methods.tex', 'v2_results.tex',
             'table_v2_common_core.tex', 'refs.bib', 'supplementary-refs.bib',
             'cas-sc.cls', 'cas-common.sty', 'cas-model2-names.bst']
    for name in names:
        shutil.copy2(SRC / name, DST / name)
    shutil.copytree(SRC / 'figs', DST / 'figs', dirs_exist_ok=True)
    for name in ['fig_v3_new_splits', 'fig_v3_boundary']:
        for ext in ['pdf', 'png']:
            shutil.copy2(OUT / f'{name}.{ext}', DST / 'figs' / f'{name}.{ext}')
    shutil.copy2(OUT / 'table_v3_new_splits.tex', DST / 'table_v3_new_splits.tex')
    direction = ('The difference of differences is positive on all three new partitions, consistent with the discovery pattern within this corpus.'
                 if notes['n_positive_interactions'] == 3 else
                 f'The difference of differences is positive on {notes["n_positive_interactions"]}/3 new partitions. The discovery pattern is therefore not uniform under new parent-grouped partitions.')
    methods = r'''\subsection{New-partition check of relative-depth benefit}\label{sec:v3design}

After the initial four-curve pilot, a follow-up plan fixed three new seeds (20261103, 20261104 and 20261105) before fitting additional models. Each split groups all wellbores sharing the same parent into one train, validation or test partition. Every seed is retained, with closed/open membership determined from source/test class coverage rather than fitted scores; all three happen to be closed. These are new partitions of the same FORCE corpus and overlap in wells. They are not an independent external dataset or independent population repetitions.

For each partition, Hybrid and Hybrid-norel are newly fitted under both curve sets, yielding twelve fits with the same full-row training budget, raw missingness masks, hyperparameters and source-well validation as before. The primary descriptive contrast is the paired difference of differences
\[
I_s=(F1_{H,4,s}-F1_{HN,4,s})-(F1_{H,16,s}-F1_{HN,16,s}),
\]
where $H$ is Hybrid and $HN$ is Hybrid-norel. Each term uses the same scored depths within its split. We report individual $I_s$, their mean and split standard deviation, without treating three overlapping partitions as a population sample. An implementation gate reproduces five v2 four-curve Hybrid seed-42 metrics to $10^{-10}$ before this schedule. Boosters, transformation state and row-level predictions are saved. One first gate attempt failed at native Windows Unicode-path persistence; a Python model-text write/readback fixes this implementation issue without changing the scientific schedule. The phase therefore has thirteen successful fits and one failed-persistence gate attempt.

\subsection{Controlled sensitivity to logged-interval boundaries}\label{sec:v3boundarydesign}

For each of the six fitted Hybrid models, let the full observed interval of a target well be $[d_{\min},d_{\max}]$ and $L=d_{\max}-d_{\min}$. Prespecified crops remove 10\% or 20\% of $L$ from the top, bottom or both. For top and bottom fractions $a,b$, retain only labelled depths in $[d_{\min}+aL,d_{\max}-bL]$ and change the relative-depth coordinate to
\[
r'={d-(d_{\min}+aL)\over (1-a-b)L+10^{-6}}.
\]
Source fitting, raw curves, missingness masks and whole-well $z$-scores remain frozen. Baseline Hybrid, changed-coordinate Hybrid and Hybrid-norel are compared on exactly the same retained rows and class denominators for each crop; the within-crop stress effect is $F1_{H,r'}-F1_{H,r}$. This prevents class-composition changes caused by cropping from being misreported as coordinate effects. Retained row/well/class counts and fixed-taxonomy tails accompany all scenarios. Models are reused for inference, so these stresses require no additional fits or test-label-based tuning. This is a controlled coordinate sensitivity analysis, not a physical incomplete-log simulation: full-curve normalisation moments are deliberately retained.
'''
    methods += '\n' + r'Boundary coordinates are reconstructed from the original float64 dataframe depths. The exported FeatureBundle depth array is float32 and is not used for this calculation. Reconstructed baseline coordinates match the trained features exactly, and saved-model baseline predictions are verified on every test row before the inference stresses. This inference recovery requires no additional model fits.' + '\n'
    (DST / 'v3_methods.tex').write_text(methods, encoding='utf-8')
    nc = bs[(bs.curve_set == 'core4') & (bs.scenario != 'baseline')]
    nf = bs[(bs.curve_set == 'full16') & (bs.scenario != 'baseline')]
    nstress = stress[stress.scenario != 'baseline']
    results = r'''\subsection{Relative-depth benefit on new parent-grouped partitions}\label{sec:v3replication}

On the three new parent-grouped partitions, adding relative depth changes mean Macro-F1 by {{GAIN4}} for four curves and {{GAIN16}} for sixteen. The mean paired difference of differences is {{INTERACTION}} (range {{IMIN}} to {{IMAX}}; individual values in Supplementary Table~S14). {{DIRECTION}} Table~\ref{tab:v3new} reports the newly fitted baseline scores, and Fig.~\ref{fig:v3new} separates the initial pilot from this follow-up. The comparison reduces dependence on the original three selected splits, while its shared corpus and overlapping wells preclude independent external-validation or population claims.

\begin{table}[width=.98\linewidth,pos=htbp]
\caption{New parent-grouped partitions, seeds 20261103/20261104/20261105. All twelve method--curve--seed rows are newly fitted, with raw missingness retained. Values are mean$\pm$sample standard deviation across three overlapping closed partitions. The cross-seed common-class set is computed within this follow-up; it need not equal the discovery set.}
\label{tab:v3new}
\centering\small
\input{table_v3_new_splits}
\end{table}

\begin{figure}[width=.96\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig_v3_new_splits.pdf}
\caption{Relative-depth benefit under two log-availability settings. Lines join the two curve-set effects within a split. The discovery and new parent-grouped phases are shown separately; they reuse the same corpus and are not independent datasets.}
\label{fig:v3new}
\end{figure}

\subsection{Sensitivity to the relative-depth coordinate}\label{sec:v3boundary}

Across the six prespecified crops, mean changed-coordinate minus baseline Macro-F1 on identical retained rows ranges from {{C4MIN}} to {{C4MAX}} for four curves and {{F16MIN}} to {{F16MAX}} for sixteen (Fig.~\ref{fig:v3boundary}). Retained-row fractions across individual split--crop comparisons range from {{KEEPLO}} to {{KEEPHI}}. The stress effect is negative in {{NEG}}/{{NSTRESS}} model--crop combinations. Supplementary Tables~S15--S16 report every seed and crop together with the matched-support Hybrid--Hybrid-norel effects. These shifts quantify sensitivity to the coordinate definition; they are not effects of removing physical measurements or recomputing all normalisation statistics from partial logs. All crop comparisons use a matched baseline, so changes in retained class coverage cannot by themselves explain a within-crop stress difference. No crop is selected as a new tuned inference policy.

\begin{figure}[width=.98\linewidth,pos=htbp]
\centering
\includegraphics[width=\linewidth]{figs/fig_v3_boundary.pdf}
\caption{Inference-only boundary-coordinate stresses. Each line is a new parent-grouped partition; black diamonds are the split means. Top/bottom/both denote which interval bounds are cropped by 10\% or 20\% of the original observed span. For each point, changed and baseline Hybrid are scored on identical retained depths. Other feature coordinates and full-curve well-normalisation moments remain fixed.}
\label{fig:v3boundary}
\end{figure}
'''
    tokens = {'GAIN4': f'${notes["mean_depth_gain_core4"]:+.3f}$',
              'GAIN16': f'${notes["mean_depth_gain_full16"]:+.3f}$',
              'INTERACTION': f'${notes["mean_difference_of_differences"]:+.3f}$',
              'IMIN': f'${inter.difference_of_differences.min():+.3f}$',
              'IMAX': f'${inter.difference_of_differences.max():+.3f}$', 'DIRECTION': direction,
              'C4MIN': f'${nc.macro_stress_delta_mean.min():+.3f}$',
              'C4MAX': f'${nc.macro_stress_delta_mean.max():+.3f}$',
              'F16MIN': f'${nf.macro_stress_delta_mean.min():+.3f}$',
              'F16MAX': f'${nf.macro_stress_delta_mean.max():+.3f}$',
              'KEEPLO': f'{nstress.fraction_rows_retained.min():.1%}'.replace('%', r'\%'),
              'KEEPHI': f'{nstress.fraction_rows_retained.max():.1%}'.replace('%', r'\%'),
              'NEG': str(int((nstress.macro_stress_delta < 0).sum())), 'NSTRESS': str(len(nstress))}
    for key, value in tokens.items():
        results = results.replace('{{' + key + '}}', value)
    bottom = nc[nc.scenario == 'bottom20'].iloc[0]
    both = nc[nc.scenario == 'both20'].iloc[0]
    reversal = ('On the matched four-curve support for the 20\\% bottom crop, the mean original '
        f'Hybrid--Hybrid-norel gain is ${bottom.original_relative_gain_mean:+.3f}$, changing to '
        f'${bottom.stressed_relative_gain_mean:+.3f}$ under the new coordinate. For the two-sided '
        f'20\\% crop the corresponding values are ${both.original_relative_gain_mean:+.3f}$ and '
        f'${both.stressed_relative_gain_mean:+.3f}$. These mean reversals illustrate the '
        'interval-definition boundary of the observed benefit within the fixed learner; they do not '
        'establish a universal ranking or a geological mechanism.\n\n')
    marker = r'\begin{figure}[width=.98\linewidth,pos=htbp]'
    results = results.replace(marker, reversal + marker, 1)
    (DST / 'v3_results.tex').write_text(results, encoding='utf-8')
    text = (DST / 'main.tex').read_text(encoding='utf-8')
    text = text.replace(r'\input{v2_methods}', r'\input{v2_methods}' + '\n\n' + r'\input{v3_methods}', 1)
    text = text.replace(r'\input{v2_results}', r'\input{v2_results}' + '\n\n' + r'\input{v3_results}', 1)
    text = text.replace('The new four-curve experiment is a bounded pilot', 'The initial four-curve experiment is a bounded pilot')
    limitation = r'The follow-up adds three parent-grouped partitions of the same corpus; it does not provide external validation or independent population replication. The coordinate-boundary stress freezes full-curve $z$-scores, so it must not be interpreted as an end-to-end incomplete-logging deployment experiment. No validated rule selects between methods on a new well.'
    text = text.replace(r'\section{Conclusions}', limitation + '\n\n' + r'\section{Conclusions}', 1)
    conclusion = ('The new parent-grouped follow-up yields a positive four-versus-sixteen-curve difference of relative-depth benefits on all three partitions, supporting an input-set-dependent effect within FORCE.'
                  if notes['n_positive_interactions'] == 3 else
                  'The new parent-grouped follow-up shows that the relative-depth pattern is not uniform across partitions; its use therefore remains conditional on the log and interval setting.')
    text = text.replace(r'\appendix', conclusion + ' The boundary-coordinate checks further quantify sensitivity to interval definitions on matched score support; they do not establish geological causality or a universal deployment policy.\n\n' + r'\appendix', 1)
    provenance = r'\paragraph{Follow-up revision artifacts.} The twelve new parent-grouped fits and boundary inference checks have a separate frozen manifest under \path{results/v3/relative_depth_validation_run2/}, with configurations, source/data/split hashes, boosters and row-level predictions. Their narrow fitting path reproduces five v2 Hybrid metrics before execution. Successful model-text files are read back and their probabilities checked on 2048 rows. These local v3 artifacts, like the v2 extension, must be included in the public revision archive before submission; no new public release DOI is claimed here.'
    text = text.replace(r'\section*{Declaration of competing interests}', provenance + '\n\n' + r'\section*{Declaration of competing interests}', 1)
    abstract = r'''
Within-well depth splitting can overstate lithofacies-classification performance on newly drilled wells. We evaluate a FORCE~2020 validation framework that separates inductive, post-drilling and causal-prefix information budgets and reports Macro-F1 together with coverage-aware fixed-taxonomy well tails. Across 18 closed-set well-held-out repetitions, inductive LightGBM achieves $0.339\pm0.045$ Macro-F1, versus 0.737 under a naive depth split; post-drilling Hybrid variants reach $0.370$--$0.379$ without uniformly improving well tails. A matched-row four- versus sixteen-curve pilot examines input availability, and an implementation audit corrects missingness-mask loss that confounded historical WellNorm comparisons. A follow-up newly fits Hybrid and Hybrid-norel on three fixed parent-grouped partitions of the same corpus. Adding relative depth yields mean Macro-F1 changes of {{GAIN4}} with four curves and {{GAIN16}} with sixteen, with a mean paired difference of {{INTERACTION}}. Controlled inference stresses crop logged-interval boundaries while keeping all other features fixed and comparing predictions on identical retained depths. These checks quantify coordinate sensitivity without claiming a physical incomplete-logging experiment. Parent-averaged analyses show heterogeneous well-level gains, and simple observable descriptors do not yet predict reversals reliably. The study links admissible information, log availability and interval definitions to the interpretation of reported scores; independent external validation and validated model-selection rules remain outstanding.
'''
    for key in ['GAIN4', 'GAIN16', 'INTERACTION']:
        abstract = abstract.replace('{{' + key + '}}', tokens[key])
    if notes['n_positive_interactions'] != 3:
        abstract = abstract.replace('Controlled inference stresses crop',
            'The direction of the paired difference varies across partitions. Controlled inference stresses crop')
    a = text.index(r'\begin{abstract}') + len(r'\begin{abstract}')
    b = text.index(r'\end{abstract}', a)
    text = text[:a] + abstract + text[b:]
    (DST / 'main.tex').write_text(text, encoding='utf-8')
    extra = [r'\FloatBarrier', '', r'\section*{New parent-grouped partitions and boundary-coordinate stresses}',
             'All three new seeds are retained and closed. These are same-corpus follow-up partitions, not external validation. Within each boundary scenario, the changed Hybrid, baseline Hybrid and Hybrid-norel share scored depths and class denominators. Delta is changed-coordinate Hybrid minus baseline Hybrid on that support. Original gain and stressed gain denote Hybrid minus Hybrid-norel before and after changing the coordinate. Other features remain frozen.', '']
    rows = [[int(r.seed), f'{r.relative_depth_delta_core4:+.4f}', f'{r.relative_depth_delta_full16:+.4f}',
             f'{r.difference_of_differences:+.4f}'] for _, r in inter.iterrows()]
    extra += [r'\begin{table}[width=.98\linewidth,pos=htbp]',
              r'\caption{Relative-depth benefit on each new parent-grouped split. All model rows are newly fitted.}',
              r'\label{tab:s-v3paired}', r'\centering\small',
              table(['Seed', 'Four-curve gain', 'Sixteen-curve gain', 'Difference of differences'], rows, 'rccc'),
              r'\end{table}', '']
    for curves in ['core4', 'full16']:
        rows = []
        for _, r in stress[stress.curve_set == curves].iterrows():
            rows.append([int(r.seed), r.scenario, int(r.n_scored),
                         f'{r.macro_stress_delta:+.3f}', f'{r.macro_relative_gain_original:+.3f}',
                         f'{r.macro_relative_gain_stressed:+.3f}', f'{r.worst_fixed_stress_delta:+.3f}'])
        extra += [r'\begin{table}[width=.98\linewidth,pos=htbp]',
                  r'\caption{' + curves + r' boundary-coordinate stresses. All contrasts use identical retained depths; worst delta uses the fixed-taxonomy screen. Full per-well and retained class counts are available in the machine-readable results.}',
                  r'\label{tab:s-v3' + curves + '}', r'\centering\small',
                  table(['Seed', 'Crop', '$n$ rows', r'$\Delta$ Macro', 'Original gain', 'Stressed gain', r'$\Delta$ worst'], rows, 'rlrcccc'),
                  r'\end{table}', '']
    supp = (DST / 'supplementary.tex').read_text(encoding='utf-8')
    supp = supp.replace(r'\end{document}', '\n'.join(extra) + '\n\n' + r'\end{document}')
    (DST / 'supplementary.tex').write_text(supp, encoding='utf-8')
    (DST / 'highlights.txt').write_text(
        'New parent-grouped partitions test relative-depth benefits under log restriction.\n'
        'Boundary-coordinate stresses compare predictions on identical scored depths.\n'
        'Frozen non-depth features isolate coordinate sensitivity without retraining.\n'
        'Saved models and row-level predictions support reproducible inference checks.\n', encoding='utf-8')
    report = ['# 新划分与相对深度边界验证', '',
              '本轮保留 V1 备份与已交付 V2，新稿单独保存到 paper/gse_v3。', '',
              '预先固定母井分组三种子 20261103、20261104、20261105，全部为闭集，全部执行。新划分上训练 12 个模型，另有 1 次成功复现；首次复现完成训练后因中文路径保存失败，因此训练尝试合计 14 次。边界检查全部复用模型。', '',
              '| 新划分种子 | 四曲线相对深度收益 | 十六曲线相对深度收益 | 配对收益差 |',
              '|---|---:|---:|---:|']
    for _, r in inter.iterrows():
        report.append(f'| {int(r.seed)} | {r.relative_depth_delta_core4:+.4f} | {r.relative_depth_delta_full16:+.4f} | {r.difference_of_differences:+.4f} |')
    report += ['', f'平均收益差为 {notes["mean_difference_of_differences"]:+.4f}；正向划分 {notes["n_positive_interactions"]}/3。这是同一 FORCE 数据集的新划分验证，不能称为独立数据集验证。', '',
               '| 曲线集 | 预设边界场景 | 保留行比例均值 | 同行评分 Macro 变化均值 | 原相对深度收益 | 扰动后相对深度收益 |',
               '|---|---|---:|---:|---:|---:|']
    for _, r in bs[bs.scenario != 'baseline'].iterrows():
        report.append(f'| {r.curve_set} | {r.scenario} | {r.retained_fraction_mean:.1%} | {r.macro_stress_delta_mean:+.4f} | {r.original_relative_gain_mean:+.4f} | {r.stressed_relative_gain_mean:+.4f} |')
    report += ['',
               '边界实验只改变相对深度的定义，保持测井值、缺失标记和整井标准化统计量不变；仅在截取区间内的相同深度比较原 Hybrid、扰动 Hybrid 与 Hybrid-norel。它能检验坐标敏感性，不能证明实际缺失井段时的完整部署表现，也不能识别地质因果机制。', '',
               '所有结果都报告，不按得分替换种子、不根据目标标签挑选边界处理方式。三个划分共享数据和部分训练井，标准差仅表示划分敏感性，没有总体显著性检验。第二独立数据集与可靠的模型选择规则仍未完成。', '',
               '模型文本通过 Python 写出，避免原生接口的中文路径限制；每个保存的模型回读后在最多 2048 行上核验概率一致。边界检查使用原始 float64 深度重建坐标，避免导出 float32 深度产生的舍入差异；重建后在全部测试行核验原模型预测一致。该恢复过程没有重训。新增边界、行支持、特征隔离、深度精度和中文路径往返测试通过，全部测试共 36 项。']
    (DST / 'REVISION_V3_CN.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Built v3 manuscript: {DST}')


if __name__ == '__main__':
    main()
