"""Create an editable scientific supplement only from the complete analysis.

Run with the bundled document Python. No measured values are embedded here.
The generated DOCX still requires render-and-visual inspection before delivery.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

METHODS = ['A', 'B', 'ref', 'A_jointmask', 'A_flipcons']
NAMES = {'A': 'A0', 'B': 'Boracle', 'ref': 'Full GT',
         'A_jointmask': 'Joint mask', 'A_flipcons': 'Flip consensus'}


def load_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def load_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def val(row, name):
    v = float(row[name])
    if not math.isfinite(v):
        raise ValueError(f'Nonfinite measured value: {name}')
    return v


def plain_font(run, size=None, bold=None):
    run.font.name = 'Times New Roman'
    run.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    for key in ('ascii', 'hAnsi'):
        rf.set(qn('w:' + key), 'Times New Roman')
    rf.set(qn('w:eastAsia'), 'Malgun Gothic')


def paragraph(doc, text, style=None):
    p = doc.add_paragraph(style=style)
    plain_font(p.add_run(text))
    return p


def caption(doc, text):
    p = paragraph(doc, text)
    p.paragraph_format.keep_with_next = True
    for run in p.runs:
        plain_font(run, size=10, bold=True)
    return p


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, w in zip(t.columns, widths):
        col.width = Inches(w)
    pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for tag in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge = OxmlElement('w:' + tag)
        for k, v in {'val': 'single', 'sz': '4', 'color': 'D9D9D9'}.items():
            edge.set(qn('w:' + k), v)
        borders.append(edge)
    pr.append(borders)
    for index, values in enumerate([headers, *rows]):
        cells = t.rows[0].cells if index == 0 else t.add_row().cells
        trpr = cells[0]._tc.getparent().get_or_add_trPr()
        no_split = OxmlElement('w:cantSplit')
        trpr.append(no_split)
        if index == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for j, (cell, text, w) in enumerate(zip(cells, values, widths)):
            cell.width = Inches(w)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side, amount in [('top', '85'), ('bottom', '85'), ('left', '95'), ('right', '95')]:
                e = OxmlElement('w:' + side)
                e.set(qn('w:w'), amount)
                e.set(qn('w:type'), 'dxa')
                margins.append(e)
            cp.append(margins)
            if index == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'EAEAEA')
                cp.append(shade)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.keep_with_next = index == 0
            plain_font(p.add_run(str(text)), size=9, bold=index == 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def add_figure(doc, file, title, explanation):
    caption(doc, title)
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(file), width=Inches(6.4))
    props = run._r.xpath('.//wp:docPr')
    for prop in props:
        prop.set('descr', explanation)
    paragraph(doc, explanation)


def build(analysis, output, protocol_path):
    status = load_json(analysis / 'analysis_status.json')
    protocol = load_json(protocol_path)
    if (status.get('status') != 'complete_planned_suite'
            or status.get('training_completed') != 120
            or status.get('external_completed') != 30
            or status.get('missing_training') or status.get('missing_external')
            or not status.get('comparability_verified')):
        raise ValueError('The complete verified 120-training/30-external matrix is required.')
    if digest(protocol_path) != status['analysis_protocol_sha256']:
        raise ValueError('Analysis protocol changed after summarization.')
    publication = load_json(analysis / 'final_publication_receipt.json')
    if (publication.get('status') != 'complete_validated'
            or publication.get('training_rows') != 120
            or publication.get('external_models') != 30):
        raise ValueError('Final publication receipt is missing or incomplete.')
    required = {'analysis_status.json', 'source_budget_curve_summary.csv', 'source_budget_auc_summary.csv',
                'penn_action_summary.csv', 'penn_action_per_seed.csv', 'penn_action_paired_clip_differences.csv',
                'source_budget_curves.png', 'source_budget_auc_per_seed.png',
                'penn_action_macro_pck_budget_0.40.png', 'penn_action_paired_differences_budget_0.40.png'}
    hashes = publication.get('artifact_sha256', {})
    if not required.issubset(hashes):
        raise ValueError('The final receipt does not cover every manuscript input.')
    for relative, expected in hashes.items():
        path = analysis / relative
        if not path.resolve().is_relative_to(analysis.resolve()) or digest(path) != expected:
            raise ValueError('Final analysis artifact changed: ' + relative)
    if output.exists():
        raise FileExistsError(f'Preserve the existing manuscript supplement: {output}')
    source = load_csv(analysis / 'source_budget_curve_summary.csv')
    auc = load_csv(analysis / 'source_budget_auc_summary.csv')
    penn = load_csv(analysis / 'penn_action_summary.csv')
    per_seed = load_csv(analysis / 'penn_action_per_seed.csv')
    differences = [r for r in load_csv(analysis / 'penn_action_paired_clip_differences.csv')
                   if r['seed'] == 'mean_of_paired_seeds']
    aindex = {r['condition']: r for r in auc}
    pindex = {r['condition']: r for r in penn}
    sindex = {(r['condition'], float(r['budget'])): r for r in source}
    if set(aindex) != set(METHODS) or set(pindex) != set(METHODS):
        raise ValueError('The five planned arms are required.')
    if any(int(r['n_training_seeds']) != 6 for r in auc + penn + source):
        raise ValueError('Each source/external summary must contain six seeds.')
    if len(per_seed) != 30 or len(differences) != 4:
        raise ValueError('Missing per-seed data or paired contrasts.')
    for row in per_seed:
        if (int(row['n_test_clips']) != 1068 or int(row['n_test_frames']) != 76298
                or int(row['n_evaluable_joints']) != 700660):
            raise ValueError('External denominator does not match the frozen test protocol.')

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(.85)
    sec.top_margin = sec.bottom_margin = Inches(.8)
    for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption']:
        st = doc.styles[name]
        st.font.name = 'Times New Roman'
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.size = Pt(11 if name == 'Normal' else 13 if name.startswith('Heading') else 18 if name == 'Title' else 10)
        st.paragraph_format.space_after = Pt(7)
        st.paragraph_format.line_spacing = 1.08
    doc.core_properties.title = 'Additional pose estimation experiments'
    doc.core_properties.subject = 'YOLO11n pseudo-label policy comparison and Penn Action external evaluation'
    doc.core_properties.author = ''
    doc.core_properties.keywords = 'pose estimation; pseudo-labeling; external evaluation; Penn Action'
    paragraph(doc, 'Additional pose estimation experiments', 'Title')
    paragraph(doc, 'Comparison of automatic pseudo label policies and external evaluation on Penn Action', 'Subtitle')
    paragraph(doc, 'This supplement extends the rehabilitation pose benchmark with two automatic pseudo-label policies and an external dataset evaluation. All planned observations are included: five methods, six training seeds, four source accepted-frame budgets, and external evaluation of all 30 checkpoints at the 40% source budget. The original A0, Boracle and full-GT observations are reused; the joint-mask and flip-consensus arms are new follow-up experiments.')

    paragraph(doc, 'Experimental design', 'Heading 1')
    paragraph(doc, 'The YOLO11n-pose backbone was trained for 30 epochs at each budget round with batch size 16 and 416-pixel inputs. Subsequent rounds cold-started from the retained pretrained backbone. Six initial GT sets were matched to the retained original A0 checkpoints and reused across the new arms. The source pool contains 8,367 frames from REHAB24-6 participants 1–8; accepted-frame targets are 418, 837, 1,673 and 3,347. The 989-frame validation set from participants 9–10 was used for checkpoint selection and remains an internal validation set with prior development exposure.')
    table(doc, ['Method', 'Additional label policy after shared 5% GT initialization'], [
        ['A0', 'Retained original self-training policy'],
        ['Boracle', 'GT-derived replacement and segmentation-metadata rejection'],
        ['Full GT', 'All newly accepted training frames use GT labels'],
        ['Joint mask', 'Relative confidence threshold; at least six retained joints'],
        ['Flip consensus', 'Joint confidence plus inverse-flip agreement and coordinate averaging'],
    ], [1.3, 5.1])
    paragraph(doc, 'The source study already included a GT-free frame-level mean-confidence gate, A_conf. The present new arms extend that comparison to joint-level visibility and horizontal-flip agreement; they are not the first confidence-based comparators for this study.')
    paragraph(doc, 'For each new-arm round, the confidence threshold is the 25th percentile of finite positive joint confidences in the remaining source training pool. It has no absolute 0.5 floor and can retain joints suppressed by the original A0 visibility conversion. Flip consensus additionally requires agreement within 5% of the image diagonal after inverse flipping and left–right remapping. Both policies reject frames with fewer than six retained joints or a degenerate retained-joint extent and acquire replacements to meet the budget. Masking also alters training-box extent. These are custom, whole-policy comparators, not reproductions of a published consistency-loss method. No new source-training GT coordinates are queried after initialization; validation GT remains available for checkpoint selection.')
    paragraph(doc, 'All original and new final checkpoints are evaluated in the same inference runtime on the official Penn Action test split without external training or tuning. Evaluation uses full RGB frames, square 640-pixel letterboxing, detection confidence 0.25, and the highest-confidence person. The 12 common limb joints exclude the Penn head point. PCK normalizes distance by the maximum side of the GT person box; missing detections count as failures. Among 76,298 frames in 1,068 videos, 174 invalid-box frames are excluded only from the PCK denominator. The common evaluable denominator is 700,660 visible joints.')

    doc.add_page_break()
    paragraph(doc, 'Source validation results', 'Heading 1')
    caption(doc, 'Table S1. Source pose mAP@50–95 and normalized label-budget AUC')
    rows = []
    for m in METHODS:
        cells = [NAMES[m]]
        for b in [.05, .1, .2, .4]:
            r = sindex[m, b]
            cells.append(f"{val(r, 'map50_95_mean'):.4f}\n± {val(r, 'training_seed_sd_ddof0'):.4f}")
        r = aindex[m]
        cells.append(f"{val(r, 'budget_auc_mean'):.4f}\n± {val(r, 'training_seed_sd_ddof0'):.4f}")
        rows.append(cells)
    table(doc, ['Method', '5%', '10%', '20%', '40%', 'AUC'], rows, [1.25, 1.03, 1.03, 1.03, 1.03, 1.03])
    paragraph(doc, 'Entries are the mean ± population SD across six fitted seeds (ddof=0), in units of 0–1. AUC uses trapezoidal integration over accepted-frame fractions 0.05–0.40, divided by 0.35. It is a source-validation label-budget measure and is distinct from external PCK. These accepted-frame budgets do not match human annotation time or human effort across arms.')
    paragraph(doc, 'External evaluation results', 'Heading 1')
    caption(doc, 'Table S2. Penn Action clip-macro PCK@0.20 at the 40% source budget')
    rows = []
    for m in METHODS:
        r = pindex[m]
        rows.append([NAMES[m], f"{val(r, 'macro_pck_020_mean') * 100:.2f}",
                     f"{val(r, 'training_seed_sd_ddof0') * 100:.2f}",
                     f"{val(r, 'conditional_clip_ci_low') * 100:.2f} to {val(r, 'conditional_clip_ci_high') * 100:.2f}"])
    table(doc, ['Method', 'Mean (%)', 'Seed SD (pp)', 'Conditional 95% CI (%)'], rows, [1.65, 1.2, 1.35, 2.2])
    paragraph(doc, 'Each video has equal weight in the primary clip-macro PCK measure. Confidence intervals use 10,000 bootstrap resamples of whole test videos after averaging the six fitted seeds within each video. They quantify video sampling uncertainty conditional on these checkpoints; they do not include training-seed or patient-population uncertainty. Training-seed SD is shown separately.')

    doc.add_page_break()
    paragraph(doc, 'External secondary metrics and paired contrasts', 'Heading 1')
    caption(doc, 'Table S3. External secondary measures at the 40% source budget')
    rows = []
    for m in METHODS:
        rr = [r for r in per_seed if r['condition'] == m]
        if len(rr) != 6:
            raise ValueError('Incomplete seed rows')
        rows.append([NAMES[m], f"{100 * mean(val(r, 'macro_pck_010') for r in rr):.2f}",
                     f"{100 * mean(val(r, 'micro_pck_020') for r in rr):.2f}",
                     f"{100 * mean(val(r, 'n_frames_with_detection') / val(r, 'n_test_frames') for r in rr):.2f}"])
    table(doc, ['Method', 'Macro PCK@0.10 (%)', 'Micro PCK@0.20 (%)', 'Detection rate (%)'], rows, [1.7, 1.57, 1.57, 1.56])
    paragraph(doc, 'Values are arithmetic means across the same six seeds. Micro PCK weights each eligible joint observation equally. Detection rate uses all 76,298 frames, including frames with invalid GT boxes for PCK.')
    caption(doc, 'Table S4. Paired differences in external clip-macro PCK@0.20')
    rows = []
    for r in differences:
        rows.append([f"Boracle minus {NAMES[r['condition_b']]}", f"{100 * val(r, 'estimate'):+.2f}",
                     f"{100 * val(r, 'ci_low'):+.2f} to {100 * val(r, 'ci_high'):+.2f}"])
    table(doc, ['Contrast', 'Difference (pp)', 'Conditional 95% CI (pp)'], rows, [2.7, 1.45, 2.25])
    paragraph(doc, 'Boracle minus flip consensus is the prospectively specified primary follow-up contrast; the remaining contrasts are exploratory. Video resampling is paired across methods. No multiplicity-adjusted significance or equivalence claim is made. Positive differences favor Boracle and negative differences favor the comparator. The comparison includes the complete adaptive label policy and its changing selection path.')
    paragraph(doc, 'Interpretation and limitations', 'Heading 1')
    primary = next(r for r in differences if r['condition_b'] == 'A_flipcons')
    paragraph(doc, f"The observed Boracle-minus-flip-consensus difference was {100 * val(primary, 'estimate'):+.2f} percentage points, with a conditional paired-video 95% interval of {100 * val(primary, 'ci_low'):+.2f} to {100 * val(primary, 'ci_high'):+.2f}. This external result must be considered together with the source-validation results and the retained per-seed measurements; a source-domain advantage alone does not establish external generalization.")
    paragraph(doc, 'Penn Action provides an external sports/action benchmark, not a rehabilitation patient cohort. Its official split is video-based; subject independence and overlap with pretrained-model training data were not established. The 12-joint full-frame protocol is not directly comparable with published 13-joint or GT-crop protocols. Boracle still uses reference coordinates, so these experiments do not establish actual clinician correction time, patient benefit or human-in-the-loop efficiency. Historical starting-weight bytes and the complete transitive software environment were not recorded; protocol comparability was audited, while historical bitwise identity remains unproven.')

    for file, title, explanation in [
        ('source_budget_curves.png', 'Figure S1. Source validation curves over accepted-frame budgets',
         'Curves show means across six seeds and error bars show population SD. The source validation set was used for checkpoint selection and had prior development exposure. Accepted-frame budgets are not human-effort budgets.'),
        ('source_budget_auc_per_seed.png', 'Figure S2. Source validation label-budget AUC for every seed',
         'Each point is one complete four-budget training-seed curve. Horizontal marks show means. Original A0, Boracle and full-GT observations are reused; the two automatic policy comparators are new follow-up experiments.'),
        ('penn_action_macro_pck_budget_0.40.png', 'Figure S3. Independent external benchmark performance',
         'Points show seed-averaged clip-macro PCK@0.20 on all 1,068 official Penn Action test videos. Intervals resample test videos conditionally on the fitted checkpoints. Seed variability is reported separately in Table S2.'),
        ('penn_action_paired_differences_budget_0.40.png', 'Figure S4. Paired external contrasts',
         'Points and intervals are differences in percentage points with conditional paired-video 95% bootstrap intervals. No patient-generalization, multiplicity-adjusted significance or equivalence claim follows from these intervals.'),
    ]:
        doc.add_page_break()
        add_figure(doc, analysis / file, title, explanation)
    paragraph(doc, 'Data and reproducibility', 'Heading 1')
    paragraph(doc, 'Penn Action official data and annotation documentation: https://dreamdragon.github.io/PennAction/. The experiment folder preserves the frozen protocol, source-validation observations, all external per-video counts and predictions, checkpoint SHA256 hashes, runtime audits, and executable analysis code. Original study materials provide REHAB24-6 acquisition and licensing details. This supplement was generated from the validated complete analysis; its tables remain editable in Word.')
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    receipt = {'status': 'created_requires_visual_qa', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'output': str(output), 'sha256': digest(output),
               'analysis_status_sha256': digest(analysis / 'analysis_status.json'),
               'protocol_sha256': digest(protocol_path), 'builder_sha256': digest(Path(__file__)),
               'training_observations': 120, 'external_models': 30, 'tables': 4, 'figures': 4}
    receipt_path = output.with_suffix('.creation.json')
    with receipt_path.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.analysis, args.output, args.protocol)
