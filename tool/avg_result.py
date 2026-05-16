import os
import glob
import pandas as pd
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3, help='GPU 번호')
    parser.add_argument('--model-csv', type=str, required=True,
                        help='v1~v5 폴더가 들어있는 디렉터리 (case 폴더)')
    parser.add_argument('--out', type=str, default=None,
                        help='출력 CSV 경로 (기본: <model-csv>/five_runs_mean_std.csv)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")
    torch.cuda.empty_cache()

    model_dir = args.model_csv

    pattern = os.path.join(model_dir, "v*", "*_results.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")

    print("Found CSVs:")
    for f in files:
        print(" -", f)

    metric_cols = ["accuracy", "precision", "recall", "f1_macro", "f1_binary"]

    dfs = []

    print("N files =", len(files))
    print("Files:")
    for f in files:
        print(" -", f)

    print("\n[Per-run Overall rows]")
    for f in files:
        df = pd.read_csv(f)
        run = os.path.basename(os.path.dirname(f))
        print(run)
        print(df[df["dataset"] == "Overall"][["dataset", "accuracy", "precision", "recall", "f1_macro", "f1_binary"]])
        df["run"] = run

        for c in metric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    expected_runs = {"v1", "v2", "v3", "v4", "v5"}
    found_runs = set(all_df["run"].unique())
    missing_runs = sorted(expected_runs - found_runs)
    if missing_runs:
        print(f"[WARN] Missing runs: {missing_runs}")

    agg = (
        all_df.groupby("dataset")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = ["dataset"] + [f"{m}_{stat}" for (m, stat) in agg.columns.tolist()[1:]]
    agg = agg.fillna(0.0)

    out_csv = args.out or os.path.join(model_dir, "five_runs_mean_std.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    agg.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    overall_rows = all_df[all_df["dataset"] == "Overall"]

    if not overall_rows.empty:
        present_runs = set(overall_rows["run"].unique())
        missing = sorted(expected_runs - present_runs)
        if missing:
            print(f"\n[WARN] Overall row is missing in runs: {missing}")
            print(f"       Overall mean/std will be computed using {len(present_runs)} runs only: {sorted(present_runs)}")

        overall = overall_rows[metric_cols].agg(["mean", "std"])
        print("\n[Overall only]")
        print(overall)
    else:
        print("\n[WARN] No Overall rows found in any run.")

if __name__ == "__main__":
    main()