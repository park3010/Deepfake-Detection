#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import hashlib
from pathlib import Path
from typing import Optional

# ========================
# EDIT THESE
# ========================
BACKBONE = "resnet50"   # or "resnet50"
STREAM = "wavelet"

TRAIN_CKPT_ROOT = Path("/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/resnet")
TEST_CSV_ROOT = Path("/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/wavelet/resnet")

OUT_AUDIT_CSV = TEST_CSV_ROOT / "audit_wavelet_eval.csv"

WAVELETS = ["haar", "sym4", "db4", "db8"]
LEVELS = [1, 2]
TYPES = ["dwt", "swt"]
SUBBANDS = ["ll", "high", "ll_energy"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_prediction_csv(out_dir: Path) -> Optional[Path]:
    matches = sorted(out_dir.glob("*_results.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] multiple result csv found in {out_dir}. Using first: {matches[0].name}")
    return matches[0]


def build_setting(backbone: str, stream: str, w: str, l: int, t: str, s: str) -> str:
    return f"{backbone}-{stream}-{w}-level{l}-{t}-{s}"


def maybe_extract_pred_counts_from_csv(pred_csv: Path):
    """
    Current test_wavelet.py result CSV does NOT contain pred_real / pred_fake.
    So return NA unless future CSV format changes.
    """
    try:
        with pred_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            if "pred_real" in fieldnames and "pred_fake" in fieldnames:
                overall_row = None
                rows = list(reader)
                for row in rows:
                    if row.get("dataset") == "Overall":
                        overall_row = row
                        break
                if overall_row is None and rows:
                    overall_row = rows[-1]

                return overall_row.get("pred_real", "NA"), overall_row.get("pred_fake", "NA")

    except Exception as e:
        print(f"[WARN] failed to inspect csv {pred_csv}: {e}")

    return "NA", "NA"


def main():
    rows = []

    for w in WAVELETS:
        for l in LEVELS:
            for t in TYPES:
                for s in SUBBANDS:
                    combo = f"{w}_level{l}_{t}_{s}"

                    run_dir = TRAIN_CKPT_ROOT / w / combo
                    best_ckpt = run_dir / f"best_{BACKBONE}_{STREAM}.pth"

                    out_dir = TEST_CSV_ROOT / w / combo
                    pred_csv = find_prediction_csv(out_dir)

                    if pred_csv is None:
                        print(f"[SKIP] no result csv: {out_dir}")
                        continue

                    if not best_ckpt.exists():
                        print(f"[WARN] result exists but checkpoint missing: {best_ckpt}")

                    csv_hash = sha256_file(pred_csv)
                    pred_real, pred_fake = maybe_extract_pred_counts_from_csv(pred_csv)

                    rows.append({
                        "Setting": build_setting(BACKBONE, STREAM, w, l, t, s),
                        "config_path": str(run_dir),
                        "checkpoint_path": str(best_ckpt),
                        "cache_path": str(out_dir),
                        "prediction_csv_path": str(pred_csv),
                        "csv_hash": csv_hash,
                        "pred_real": pred_real,
                        "pred_fake": pred_fake,
                    })

    OUT_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_AUDIT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Setting",
                "config_path",
                "checkpoint_path",
                "cache_path",
                "prediction_csv_path",
                "csv_hash",
                "pred_real",
                "pred_fake",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] audit csv saved to: {OUT_AUDIT_CSV}")
    print(f"[OK] rows: {len(rows)}")


if __name__ == "__main__":
    main()