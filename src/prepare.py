import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare stroke dataset")
    parser.add_argument("--input_path", type=str, default="data/raw/healthcare-dataset-stroke-data.csv")
    parser.add_argument("--output_dir", type=str, default="data/prepared")
    parser.add_argument("--target_col", type=str, default="stroke")
    parser.add_argument("--drop_cols", type=str, default="id", help="Comma-separated columns to drop")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_path = Path(args.input_path)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()

    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Columns: {df.columns.tolist()}")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.target_col]
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Data preparation completed successfully.")
    print(f"Input file:  {input_path}")
    print(f"Train file:  {train_path}")
    print(f"Test file:   {test_path}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")


if __name__ == "__main__":
    main()