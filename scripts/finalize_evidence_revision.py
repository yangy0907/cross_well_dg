"""Write the requested self-check and package the evidence-linked revision."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / 'paper/gse_evidence_20261104'
OUT = ROOT / 'submission_evidence_20261104'


def self_check():
    audit = json.loads((DST / 'EVIDENCE_STYLE_AUDIT.json').read_text(encoding='utf-8'))
    aux = (DST / 'main.aux').read_text(encoding='utf-8')
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
    def ref(key):
        assert key in labels, key
        return ('Table ' if key.startswith('tab:') else 'Figure ' if key.startswith('fig:') else 'Section ') + labels[key]
    claims = [
        ('We define inductive, post-drilling and causal-prefix information settings.', ref('fig:method') + '; ' + ref('tab:methods'), 'Definitions and implemented configurations; not an algorithm-superiority claim.'),
        ('The reporting record links metrics to information access and training counts.', ref('tab:checklist') + '; ' + ref('tab:accept'), 'Implemented reporting output.'),
        ('Global-LGBM achieves Macro-F1 0.339 ± 0.045 over 18 closed well-id splits.', ref('tab:multirep'), 'Mean and sample SD over overlapping FORCE partitions.'),
        ('Hybrid achieves 0.379 ± 0.037; HN achieves 0.370 ± 0.044.', ref('tab:multirep'), 'Post-drilling information setting, compared with the explicitly identified inductive baseline.'),
        ('Hybrid exceeds HN in mean Macro-F1 but has lower Worst-fixed on well-id splits.', ref('tab:multirep') + '; ' + ref('fig:pareto'), 'Observed 0.379/0.370 and 0.101/0.105 trade-off; tails include coverage.'),
        ('Parent grouping changes the mean Hybrid-minus-HN Worst-fixed difference from −0.004 to +0.006.', ref('tab:pairedmr') + '; ' + ref('fig:paretoparent'), 'Observed descriptive grouping sensitivity.'),
        ('Naive depth splitting yields 0.737, versus 0.497 under contiguous depth blocks.', ref('tab:matched'), 'Separate test supports; not a pure leakage effect.'),
        ('Matched target-row diagnostics yield +0.028 and published equal-row +0.024.', ref('tab:matched'), 'Training composition changes; no additive causal decomposition.'),
        ('The sibling-removal diagnostic changes Macro-F1 from 0.326 to 0.324.', 'Supplementary Table S7', 'One seed and one sibling removal.'),
        ('Restricting inputs to four curves reduces the pilot mean for all four methods.', ref('tab:v2core') + '; ' + ref('fig:v2core'), 'Matched labelled rows, three overlapping closed pilot splits.'),
        ('The pilot four-minus-sixteen difference of relative-depth gains is +0.035.', ref('tab:v2core') + '; ' + ref('fig:v3new'), 'Initial three-split descriptive contrast.'),
        ('The new-partition difference averages +0.011 and is positive in 2/3 partitions.', ref('tab:v3new') + '; ' + ref('fig:v3new') + '; Supplementary Table S14', 'Three new parent-grouped partitions of the same corpus.'),
        ('Coordinate stresses reduce four-curve mean Macro-F1 by 0.011–0.052.', ref('fig:v3boundary') + '; Supplementary Tables S15–S16', 'Same retained rows, fixed other features and full-curve normalisation.'),
        ('Bottom/two-sided 20% stresses reverse the mean four-curve HN comparison.', 'Supplementary Table S15; ' + ref('fig:v3boundary'), 'Original +0.035/+0.034 versus stressed −0.017/−0.018 on matched support.'),
        ('Restoring raw masks changes seed-42 WellNorm Macro-F1 from 0.293 to 0.356.', 'Supplementary Table S12; ' + ref('sec:v2pilot'), 'Same eight-thread diagnostic; three-split mask gains are not uniform.'),
        ('HN gains vary across parent-averaged wells.', ref('fig:v2heterogeneity') + '; ' + ref('sec:v2heterogeneity'), 'Repeated well appearances averaged before parents; shared training induces dependence.'),
        ('The tested descriptors have weak rank associations with paired gain.', ref('fig:v2heterogeneity') + '; released parent-level summaries', 'Spearman associations, not causal explanations or a validated prediction rule.'),
        ('Chalk and Dolomite remain near-zero F1; Tuff is frequently assigned to Shale.', ref('fig:perclass') + '; ' + ref('fig:rarecm'), 'Seed-42 class-level diagnostic; no calibrated triage guarantee.'),
        ('Nearly all seed-42 scored depths have at least 100 prefix samples.', 'Supplementary Table S1', 'Completed-wireline prefix support; not sparse-prefix or MWD/LWD validation.'),
        ('Historical CPU fits have median runtime 211 s and IQR 174–256 s.', 'Result CSV seconds fields; Computational cost subsection', 'Historical workstation schedule; no comparison with eight-thread follow-up fits.'),
    ]
    prose_reduction = 100 * (1 - audit['prose_tokens_after'] / audit['prose_tokens_before'])
    text = [
        '# Evidence-linked revision: self-check', '', 'Completed 4 November 2026.', '',
        'This revision applies the requested English writing rules to the existing FORCE manuscript. It changes wording, evidence placement and section organisation. It adds no experiments, data, citations or scientific conclusions.', '',
        '## 1. Hedging and voice', '',
        'The specified word-and-phrase scan identified five matches in the preceding version. All five were reviewed in context:', '',
        '| Original match | Classification | Action |', '|---|---|---|',
        '| may (two occurrences) | Defined permission to use unlabelled target curves, not uncertainty | Replace with permits during training. |',
        '| it appears | Literal occurrence of the unk class in test truth, not uncertainty | Replace with test truth contains it. |',
        '| relatively | Vague class-performance assessment | Replace with class-specific observations and figure references. |',
        '| could | Conditional competing-interest wording, not a scientific hedge | Retain no known interests and express actual or perceived influence without a modal verb. |', '',
        'The final manuscript and Supplementary sources contain zero matches to the specified banned words and phrases. Genuine uncertainty is represented through the actual experimental conditions, split-to-split ranges, sign variation and named validation requirements rather than vague possibility words.', '',
        'An extended modal scan retains three non-hedging uses: would describes a different full-grid filtering procedure, should introduces a reporting-checklist instruction, and can states a fixed-taxonomy metric property under an explicit k≪K condition with an observed well example. Their contexts and classifications are recorded in EVIDENCE_STYLE_AUDIT.json.', '',
        'We replaced agentless fitting and reporting constructions with explicit actors in the revised Methods and Results passages. Definition-focused passive constructions and unchanged table wording remain where the object, rather than the actor, carries the technical meaning. We did not insert outperform or eliminate where the comparisons do not establish those claims.', '',
        '## 2. Paragraph and section logic', '',
        'The opening of each narrative paragraph was reviewed for its role: background claim, method definition, measured result, interpretation or validation requirement. Core empirical paragraphs now lead with the finding and follow with its number or figure/table anchor. Methods paragraphs state the executed procedure before its conditions. The primary Results subsection starts with the 18-split performance result; mask provenance follows the finding.', '',
        'Discussion is now a separate main section. Limitations and future work form a subsection within it. Each limitation names the comparison it restricts and the validation needed to resolve that restriction. Necessary conditions remain beside the corresponding claims: overlapping partitions, same retained depths, fixed other features, class-coverage denominators and corrected versus legacy masks.', '',
        '## 3. Abstract check', '',
        '| Required move | Current content |', '|---|---|',
        '| Background gap | Scores alone do not specify target-well information access or log availability. |',
        '| Method | Information-budget-aware FORCE framework, fixed LightGBM family, matched-row curve study and controlled coordinate stresses. |',
        '| Quantified results | 0.339 ± 0.045 versus 0.737; Hybrid 0.370–0.379; paired gain difference +0.011; coordinate-stress losses 0.011–0.052. |',
        '| Significance | Scores become traceable to the target information, available curves and interval coordinate used by the classifier. |', '',
        'The abstract contains no generic self-limitation or future-work sentence and no banned hedge. It retains measured heterogeneity and the experimental conditions required to interpret the results. Negative or mixed results are findings, not removable limitations.', '',
        '## 4. Claim–evidence check', '',
        '| Core claim | Immediate evidence | Scope preserved |', '|---|---|---|',
    ]
    text += ['| ' + ' | '.join(row) + ' |' for row in claims]
    text += ['',
        'The four contribution items have explicit evidence anchors: configuration/framework outputs, the 18-split score/tail comparison, the matched curve and coordinate studies, and the split/well provenance records. We make no universal curve-count, algorithm-superiority, geological-mechanism or calibrated-deployment claim.', '',
        '## 5. Main-text discipline and consistency', '',
        f'- Revised units: {audit["edited_units"]}. REWRITE_COMPARISON.md records all actual edits as original → revised → reason; REWRITE_EDITS.json preserves the exact source strings.',
        f'- Approximate narrative token count: {audit["prose_tokens_before"]:,} → {audit["prose_tokens_after"]:,}, a {prose_reduction:.1f}% reduction in this round. This repeats the preceding audit metric and is not a journal word count.',
        '- Kept: primary means/SDs, paired effects, sign variation, negative findings, information-access definitions and matched-support conditions.',
        '- Compressed or deleted: duplicate introductory framing, subjective performance descriptions, unsupported causal/ceiling explanations and repeated algorithm-ranking disclaimers.',
        '- Relocated: broad deployment and inference limitations into Discussion, each paired with a concrete next validation. No new SI experiments or numerical tables were added.',
        '- Protected: all table environments, display equations, per-file citation-key sets, bibliography files and figure assets. One Pareto caption now reports the observed −0.004 to +0.006 change instead of can flip; the plotted data are unchanged.',
        '- Terminology: Global-LGBM, Hybrid-norel (HN), Hybrid, Hybrid-while, corrected WellNorm, Macro-F1, Worst-fixed and W10 retain their established meanings. Sample SDs describe overlapping-split sensitivity, not population confidence.',
        '- Implementation wording: missingness indicators precede median imputation; the corrected pilot reads the raw observed channel. This clarifies the existing implementation in src/features.py and does not change features or results.',
        '', '## 6. Files and reproducibility', '',
        'Use main.pdf and supplementary.pdf in this directory as the current wording. The previous gse_polished_20261103 and gse_final_20261103 versions remain unchanged.', '',
        'The separate GSE_evidence_revision_20261104.zip package contains the current PDFs, editable LaTeX, figures, bibliography, rewrite comparison and audit files. The earlier GSE_review_materials_20261103.zip remains the code/data/result archive for the same experiments; its manuscript wording predates this revision. No new public deposition or submission is asserted.', '',
        'Compilation and page-render verification are recorded separately in qa/final_check.json.',
    ]
    (DST / 'SELF_CHECK.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    (DST / 'CLAIM_EVIDENCE_MATRIX.json').write_text(json.dumps([dict(zip(['claim', 'evidence', 'scope'], row)) for row in claims], indent=2, ensure_ascii=False), encoding='utf-8')


def package():
    qa = json.loads((DST / 'qa/final_check.json').read_text(encoding='utf-8'))
    assert qa['visual_review'] == 'passed'
    OUT.mkdir(exist_ok=True)
    files = [p for p in DST.iterdir() if p.is_file() and p.suffix in {'.tex', '.bib', '.cls', '.sty', '.bst', '.pdf', '.md', '.txt', '.json'} and not p.name.endswith('_build.json')]
    files += [p for p in (DST / 'figs').rglob('*') if p.is_file()]
    files += [DST / 'qa/final_check.json']
    manifest = {p.relative_to(DST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    target = OUT / 'GSE_evidence_revision_20261104.zip'
    readme = '''Current evidence-linked English revision, 2026-11-04.
Use manuscript/main.pdf and manuscript/supplementary.pdf.
REWRITE_COMPARISON.md contains original/revised/reason entries.
SELF_CHECK.md documents the requested writing self-check and claim/evidence audit.
Editable LaTeX, bibliography and figures are in manuscript/.
Compile from manuscript/: latexmk -pdf main.tex; latexmk -pdf supplementary.tex

No new experiments were run. The original GSE_review_materials_20261103.zip
remains the separate research code/data/result archive. Its manuscript wording
predates this revision. Earlier final and language-only versions are preserved.
'''
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt', readme)
        z.writestr('MANIFEST_SHA256.json', manifest_text)
        for p in sorted(files):
            z.write(p, 'manuscript/' + p.relative_to(DST).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read('manuscript/' + name)).hexdigest() == expected
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n', encoding='utf-8')
    (OUT / 'MANIFEST_SHA256.json').write_text(manifest_text, encoding='utf-8')
    print(json.dumps({'archive': str(target), 'files': len(files), 'bytes': target.stat().st_size, 'sha256': digest, 'archive_verification': 'passed'}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.package:
        package()
