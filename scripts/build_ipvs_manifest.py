"""Join Italian PVS recordings to their original age/sex metadata.

Exact normalised name + surname matches only; unmatched files are audited.
No age decoding from filenames. Task codes come from FILE CODES.xlsx.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
import pandas as pd
from scripts.inspect_ipvs_metadata import rows


def key(name):
    return "".join(c for c in unicodedata.normalize("NFKD", name).casefold() if c.isalnum())


def build_manifest(root: Path, output: Path, task="sustained_vowel"):
    records, unmatched = [], []
    groups = [("15 Young Healthy Control", "15 YHC.xlsx", 0, ("A", "B", "C", "D")),
              ("22 Elderly Healthy Control", "Tab 3.xlsx", 0, ("A", "B", "C", "D")),
              ("28 People with Parkinson's disease", "TAB 5.xlsx", 1, ("B", "C", "D", "E"))]
    for group, workbook, label, columns in groups:
        folder = root / group
        metadata = {}
        ambiguous = set()
        for row in rows(folder / workbook):
            cells = {re.sub(r"\d+$", "", cell): value for cell, value in row.items()}
            name, surname, sex, age = [cells.get(c, "").strip() for c in columns]
            if not age.isdigit() or not name or not surname or sex not in {"M", "F"}:
                continue
            identity = key(name + surname)
            if identity in metadata:
                ambiguous.add(identity)
            metadata[identity] = {"age": int(age), "sex": sex,
                "metadata_source": str((folder / workbook).resolve()), "metadata_row": next(iter(row))}
        for recording in sorted(folder.rglob("*.wav")):
            # Include only sustained /a/ or phonemically balanced text B1/B2.
            codes = ("VA1", "VA2") if task == "sustained_vowel" else ("B1", "B2")
            if not recording.name.upper().startswith(codes):
                continue
            identity = key(recording.parent.name)
            if identity not in metadata or identity in ambiguous:
                unmatched.append({"path": str(recording.relative_to(root)),
                    "reason": "ambiguous_metadata_identity" if identity in ambiguous else "no_exact_metadata_match"})
                continue
            records.append({"path": str(Path(os.path.relpath(recording, output.parent))),
                "speaker_id": f"ipvs:{label}:{identity}", "label": label,
                "task": task, **metadata[identity]})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No exact metadata matches; inspect the dataset layout.")
    frame.to_csv(output, index=False)
    audit = {"matched_recordings": len(frame), "matched_speakers": int(frame.speaker_id.nunique()),
             "eligible_recordings": int((frame.age >= 50).sum()),
             "eligible_speakers": int(frame[frame.age >= 50].speaker_id.nunique()),
             "unmatched_recordings": unmatched, "task": task,
             "source": "Original Italian PVS workbooks; exact normalised name/surname joins."}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ItalianPVS"))
    parser.add_argument("--output", type=Path, default=Path("data/ipvs_manifest.csv"))
    parser.add_argument("--task", choices=["sustained_vowel", "reading"], default="sustained_vowel")
    args = parser.parse_args()
    audit = build_manifest(args.input.resolve(), args.output.resolve(), args.task)
    print(json.dumps({k: v for k, v in audit.items() if k != "unmatched_recordings"}, indent=2))
    print(f"Unmatched recordings: {len(audit['unmatched_recordings'])}; see audit JSON.")


if __name__ == "__main__":
    main()
