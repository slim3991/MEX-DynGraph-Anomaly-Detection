import mlflow
import pandas as pd


def params_latex(df: pd.DataFrame):
    df["params.rank"] = df["params.rank"].combine_first(df["params.ranks"])

    # Optionally drop the redundant column
    df = df.drop(columns=["params.ranks"])

    cols_to_keep = [
        "params.name",
        "params.rank",
        "params.local_threshold",
        "params.threshold",
        "params.lambdas",
        "params.ks",
    ]
    available_cols = [c for c in cols_to_keep if c in df.columns]

    if not available_cols:
        print("No runs found with the specified metrics.")
    else:
        filtered_df = df[available_cols].copy()

        # 4. Clean up column names for LaTeX
        filtered_df.columns = [
            c.split(".")[-1].replace("_", " ").title() for c in filtered_df.columns
        ]

        # 5. Generate LaTeX
        latex_code = filtered_df.to_latex(
            index=False,
            caption="optimal metrics found during training",
            label="tab:eval-train-parameters",
            float_format="%.4f",
            column_format="l" + "c" * (len(filtered_df.columns) - 1),
        )
        print(latex_code)


def metric_latex(df):
    cols_to_keep = [
        "params.name",
        "metrics.f1",
        "metrics.pr_auc",
        "metrics.recall",
        "metrics.precision",
        "metrics.events_tpr",
        "metrics.events_score",
    ]

    # Keep only columns that exist AND have at least one non-null value
    available_cols = [
        c for c in cols_to_keep if c in df.columns and df[c].notna().any()
    ]

    if not available_cols:
        print("No runs found with the specified metrics.")
        return

    filtered_df = df[available_cols].copy()

    # Clean up column names for LaTeX
    filtered_df.columns = [
        c.split(".")[-1].replace("_", " ").title() for c in filtered_df.columns
    ]

    latex_code = filtered_df.to_latex(
        index=False,
        caption=("Average of 10 evaluation with random anomalies. train"),
        label="tab:eval-train-rand-anomalies",
        float_format="%.4f",
        column_format="l" + "c" * (len(filtered_df.columns) - 1),
    )

    print(latex_code)


def main():
    tag = "32775a13"
    df = mlflow.search_runs(
        filter_string=f"tags.eval_run = '{tag}' AND metrics.f1 >=0",
        search_all_experiments=True,
    )
    params_latex(df)
    metric_latex(df)


if __name__ == "__main__":
    main()
