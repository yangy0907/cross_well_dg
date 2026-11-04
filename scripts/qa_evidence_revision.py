"""Render checks for the evidence-linked manuscript version."""
from pathlib import Path
import qa_language_revision as qa

ROOT = Path(__file__).resolve().parents[1]
qa.SRC = ROOT / 'paper/gse_polished_20261103'
qa.DST = ROOT / 'paper/gse_evidence_20261104'
qa.QA = qa.DST / 'qa'

if __name__ == '__main__':
    qa.main()
