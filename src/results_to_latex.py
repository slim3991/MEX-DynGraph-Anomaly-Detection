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


import yaml


def generate_latex_tables(yaml_content):
    data = yaml.safe_load(yaml_content)
    tables = []

    for category, models in data.items():
        title = category.replace("_", " ").title()

        # Table Header
        latex_str = [
            f"% Table for {title}",
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{Model Parameters for {title}}}",
            "\\begin{tabular}{l l l}",
            "\\hline",
            "\\textbf{Model} & \\textbf{Parameter} & \\textbf{Value} \\\\",
            "\\hline",
        ]

        for model_name, params in models.items():
            # Clean underscores for LaTeX
            display_name = f"\\textbf{{{model_name.replace('_', ' ')}}}"

            first_row = True
            for param_key, param_value in params.items():
                # Format model name only on the first line of its parameter block
                model_col = display_name if first_row else ""

                # Format values (lists vs single numbers)
                val_str = str(param_value).replace("_", "\\_")
                param_label = param_key.replace("_", " ").capitalize()

                latex_str.append(f"{model_col} & {param_label} & {val_str} \\\\")
                first_row = False

            latex_str.append("\\hline")

        latex_str.append("\\end{tabular}")
        latex_str.append("\\end{table}\n")

        tables.append("\n".join(latex_str))

    return tables


def main():
    # Your data string

    with open("src/model_config.yaml", "r") as f:
        yaml_input = f.read()

    print(generate_latex_tables(yaml_input)[1])
    exit()
    tag = "bd226d7c"
    df = mlflow.search_runs(
        filter_string=f"tags.eval_run = '{tag}' AND metrics.f1 >=0",
        search_all_experiments=True,
    )
    params_latex(df)
    metric_latex(df)


if __name__ == "__main__":
    main()
