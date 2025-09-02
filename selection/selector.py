import ast
import json
import math
import os
from typing import Any

import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import shap

from tqdm import tqdm

from flink.knob import KnobDef, parse_knob_def
from utils.config import RootConfig, load_full_knob_root_config

importance_method = "null"  # null, permutation, or shap


def calculate_permutation_importance(
    fitted_model, x_test, y_test, parameter_cols, rand
):
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        fitted_model,
        x_test,
        y_test,
        n_repeats=50,
        random_state=rand.integers(0, 10000),
        n_jobs=-1,
    )
    # Use dict-style access to please strict type checkers
    importance_scores = result["importances_mean"] if isinstance(result, dict) else result.importances_mean
    return {param: score for param, score in zip(parameter_cols, importance_scores)}


def calculate_null_importance(fitted_model, x_train, y_train, parameter_cols, rand):
    feature_importances = fitted_model.feature_importances_
    null_importances = []
    for _ in range(50):  # Repeat multiple times for stability
        y_shuffled = y_train.sample(
            frac=1, random_state=rand.integers(0, 10000)
        ).reset_index(drop=True)
        rf_null = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=1,
            max_features=1,
            random_state=rand.integers(0, 10000),
            n_jobs=-1,
        )
        rf_null.fit(x_train, y_shuffled)
        null_importances.append(rf_null.feature_importances_)

    null_importances = np.array(null_importances)
    mean_null_importances = null_importances.mean(axis=0)
    std_null_importances = null_importances.std(axis=0)
    # Add small value to avoid division by zero
    importance_scores = (feature_importances - mean_null_importances) / (
        std_null_importances + 1e-9
    )
    param_importance_dict = {
        param: score for param, score in zip(parameter_cols, importance_scores)
    }
    return dict(
        sorted(param_importance_dict.items(), key=lambda item: item[1], reverse=True)
    )


def calculate_shap_importance(
    fitted_model, X: pd.DataFrame, y, parameter_cols: list[str], rand
):
    """
    Compute feature importances using SHAP values.
    """
    try:
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X)
    except Exception:
        explainer = shap.Explainer(fitted_model, X)
        shap_values = explainer(X)

    # Normalize to a single 2D array of shape (n_samples, n_features)
    def _to_2d(vals):
        try:
            import numpy as _np
            if hasattr(vals, "values"):
                # Explanation object
                return _np.asarray(vals.values)
        except Exception:
            pass
        if isinstance(vals, list) and len(vals) > 0:
            arrays = [np.asarray(v) for v in vals]
            abs_mean = np.mean([np.abs(a) for a in arrays], axis=0)
            return abs_mean
        return np.asarray(vals)

    sv_2d = _to_2d(shap_values)
    # sv_2d shape: (n_samples, n_features). Compute mean absolute across samples.
    mean_abs_importance = np.mean(np.abs(sv_2d), axis=0)

    param_importance = {
        param: float(score) for param, score in zip(parameter_cols, mean_abs_importance)
    }
    # Return sorted (high to low) for readability
    return dict(sorted(param_importance.items(), key=lambda kv: kv[1], reverse=True))



def parse_selection_json(
    filename: str,
) -> tuple[list[str], DataFrame, float, float]:
    # Example file: [[{"key1": value1, ...}, obj_1, ignore, obj_2], ...]]
    with open(filename, "r") as f:
        data = json.load(f)
    knob_names = list(data[0][0].keys())
    knob_values = [list(item[0].values()) for item in data]
    objs = [(item[1], item[3]) for item in data]
    if len(knob_values) != len(objs):
        raise ValueError("Number of knob values does not match number of objectives.")
    non_zero_count = 0
    sum_1, sum_2 = 0, 0
    for obj in objs:
        if obj[0] > 0 and obj[1] > 0:
            sum_1 += obj[0]
            sum_2 += obj[1]
            non_zero_count += 1

    avg_1 = sum_1 / non_zero_count if non_zero_count > 0 else 0
    avg_2 = sum_2 / non_zero_count if non_zero_count > 0 else 0
    # Calculate weights
    weight_1 = 1 / avg_1 if avg_1 > 0 else 1
    weight_2 = 1 / avg_2 if avg_2 > 0 else 1
    print("weight_1:", weight_1)
    print("weight_2:", weight_2)
    # 1. Create dataframe
    df = pd.DataFrame(knob_values, columns=knob_names)
    obj_orig_1 = pd.DataFrame([obj[0] for obj in objs], columns=["throughput_orig"])
    obj_orig_2 = pd.DataFrame([obj[1] for obj in objs], columns=["resource_orig"])
    obj_1 = pd.DataFrame([weight_1 * obj[0] for obj in objs], columns=["throughput"])
    obj_2 = pd.DataFrame([weight_2 * obj[1] for obj in objs], columns=["resource"])
    y = pd.DataFrame(
        [
            (
                -2.0
                if math.isclose(obj[0], 0) and math.isclose(obj[1], -1)
                else weight_1 * obj[0] - weight_2 * obj[1]
            )
            for obj in objs
        ],
        columns=["performance"],
    )
    df = pd.concat([df, obj_orig_1, obj_orig_2, obj_1, obj_2, y], axis=1)
    # 2. Data Preprocessing
    # 2.1 Handle FAILED values and missing data
    for col in ["performance"]:
        # df = df[df[col] != "FAILED"]  # Remove rows where performance is FAILED
        # df = df[~df[col].isna()]  # Remove rows where performance is nan
        # df = df[df[col] > 0]
        df[col] = df[col].astype(float)  # Ensure performance metrics are float

    return knob_names, df, weight_1, weight_2


def parse_selection_log(
    filename: str,
) -> tuple[list[str], DataFrame, float, float]:
    """
    Parses the selection log file and returns a list of tuples containing
    knob values and their corresponding performance metrics.

    Args:
        filename (str): The path to the selection log file.

    Returns:
        knob values list, objectives list, objective 1 weight, objective 2 weight
    """
    # Example lines:
    # Knob values: {'jobmanager.memory.enable-jvm-direct-memory-limit': ...}
    # Objs: [38.44647777777778, 1207.072691552063]
    knob_values: list[list[Any]] = []
    objs: list[list[float]] = []
    knob_names = []
    units = ["b", "kb", "mb", "gb", "ms", "s", "min", "h"]

    current_knob_values = None

    with open(filename, "r") as f:
        for line in f:
            if "Knob values: " in line:
                # Extract knob values from the line
                string = line.split(": ", 1)[1]
                # Remove units
                for unit in units:
                    string = string.replace(" " + unit, "")
                knob_dict = yaml.safe_load(string)
                if not knob_names:
                    knob_names = list(knob_dict.keys())
                else:
                    # Ensure the knob names are consistent
                    if list(knob_dict.keys()) != knob_names:
                        raise ValueError("Knob names do not match across samples.")
                current_knob_values = []
                for knob_name in knob_names:
                    if knob_name in knob_dict:
                        current_knob_values.append(knob_dict[knob_name])
                    else:
                        raise ValueError(f"Knob {knob_name} not found in the log.")

            elif "Objs: " in line:
                # Extract performance metric from the line
                current_objs = json.loads(line.split(": ")[1])
                if current_knob_values is not None:
                    knob_values.append(current_knob_values)
                    objs.append(current_objs)
                    current_knob_values = None
    if len(knob_values) != len(objs):
        raise ValueError

    non_zero_count = 0
    sum_1, sum_2 = 0, 0
    for obj in objs:
        if obj[0] > 0 and obj[1] > 0:
            sum_1 += obj[0]
            sum_2 += obj[1]
            non_zero_count += 1

    avg_1 = sum_1 / non_zero_count if non_zero_count > 0 else 0
    avg_2 = sum_2 / non_zero_count if non_zero_count > 0 else 0
    # Calculate weights
    weight_1 = 1 / avg_1 if avg_1 > 0 else 1
    weight_2 = 1 / avg_2 if avg_2 > 0 else 1
    print("weight_1:", weight_1)
    print("weight_2:", weight_2)

    # 1. Create dataframe
    df = pd.DataFrame(knob_values, columns=knob_names)
    obj_orig_1 = pd.DataFrame([obj[0] for obj in objs], columns=["throughput"])
    obj_orig_2 = pd.DataFrame([obj[1] for obj in objs], columns=["resource"])
    # Apply weights to objectives
    obj_1 = pd.DataFrame([weight_1 * obj[0] for obj in objs], columns=["throughput"])
    obj_2 = pd.DataFrame([weight_2 * obj[1] for obj in objs], columns=["resource"])
    y = pd.DataFrame(
        [weight_1 * obj[0] + weight_2 * obj[1] for obj in objs], columns=["performance"]
    )
    df = pd.concat([df, obj_orig_1, obj_orig_2, obj_1, obj_2, y], axis=1)

    # 2. Data Preprocessing
    # 2.1 Handle FAILED values and missing data
    for col in ["performance"]:
        # df = df[df[col] != "FAILED"]  # Remove rows where performance is FAILED
        # df = df[~df[col].isna()]  # Remove rows where performance is nan
        # df = df[df[col] > 0]
        df[col] = df[col].astype(float)  # Ensure performance metrics are float

    return knob_names, df, weight_1, weight_2


def plot_param_performance(df: pd.DataFrame, knob_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    perf_col = df.columns[-1]
    param_cols = df.columns[:-1]
    for knob_name in tqdm(knob_names):
        x = df[knob_name]
        y = df[perf_col]
        # 判断是否离散变量（字符串、布尔等）
        if pd.api.types.is_numeric_dtype(x):
            # 连续值：原样绘制
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y)
            plt.xlabel(knob_name)
            plt.ylabel(perf_col)
            plt.title(f"{knob_name} vs {perf_col}")
        else:
            # 离散值：按排序后编号作横坐标
            labels = sorted(x.dropna().unique(), key=str)
            value_to_num = {v: i for i, v in enumerate(labels)}
            x_numeric = x.map(value_to_num)
            plt.figure(figsize=(6, 4))
            plt.scatter(x_numeric, y, alpha=0.5)
            plt.xticks(range(len(labels)), labels, rotation=20)
            plt.xlabel(knob_name)
            plt.ylabel(perf_col)
            plt.title(f"{knob_name} vs {perf_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{knob_name}.png"))
        plt.close()


def analyze_parameter_importance(knob_names, df):
    """
    Analyzes parameter importance for each performance metric in an XLSX file.

    Args:
        file_path (str): The path to the XLSX file.

    Returns:
        dict: A dictionary where keys are performance metric names and values are
              dictionaries of parameter importances.
    """

    # 2.2 Handle different parameter types (enum, bool, int, float)
    encoders = {}  # Store encoders for each categorical column
    for col in knob_names:
        if df[col].dtype == "object":  # Handle string/enum types
            # Use OrdinalEncoder to handle potential unseen values in test data
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[col] = enc.fit_transform(df[[col]])
            encoders[col] = enc
        elif df[col].dtype == "bool":
            df[col] = df[col].astype(int)  # Convert boolean to int (0 and 1)
        # Int and float columns don't need special handling

    # 3. Parameter Importance Analysis (for each performance metric)
    results: dict[str, dict[str, float]] = {}
    for metric_col in ["throughput", "resource", "performance"]:
        # 3.1 Prepare data for the current metric
        X = df[knob_names]
        y = df[metric_col]

        # 3.2 Split data (optional but recommended for more robust importance estimation)
        rand = np.random.Generator(np.random.PCG64(42))
        # x_train, x_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=rand.integers(0, 10000)
        # )

        # 3.3 Train Random Forest and get feature importances
        rf_model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=1,
            max_features=1,
            random_state=rand.integers(0, 10000),
            n_jobs=-1,
        )
        rf_model.fit(X, y)

        if importance_method == "permutation":
            results[metric_col] = calculate_permutation_importance(
                rf_model, X, y, knob_names, rand
            )
        elif importance_method == "null":
            results[metric_col] = calculate_null_importance(
                rf_model, X, y, knob_names, rand
            )
        elif importance_method == "shap":
            results[metric_col] = calculate_shap_importance(
                rf_model, X, y, knob_names, rand
            )

    return results


def select_knobs(
    knob_names: list[str],
    df: DataFrame,
    importances: dict[str, float],
    full_knob_conf: list[KnobDef],
    num_knobs: int = 40,
):
    """
    Selects knobs based on their importance scores.
    """
    # From the knob names, select knobs with top 10 performance in df
    # Then print the KnobDef of these knobs in yaml format
    sorted_importances = sorted(
        importances.items(), key=lambda item: item[1], reverse=True
    )
    sorted_importances = [
        (name, importances[name])
        for name, _ in sorted_importances
        if name in knob_names
    ]
    selected_knobs = [name for name, _ in sorted_importances[:num_knobs]]
    print("Selected knobs:", selected_knobs)
    print("Performances:", [importances[name] for name in selected_knobs])
    print("Knob Definitions:")

    selected_knob_defs = [
        knob for knob in full_knob_conf if knob.name in selected_knobs
    ]
    lines = [
        f"    {line}"
        for line in yaml.dump(
            [knob.to_dict() for knob in selected_knob_defs], sort_keys=False
        ).splitlines()
    ]
    for line in lines:
        print(line)


def run(conf: RootConfig):
    import pandas as pd

    knob_selection_conf = load_full_knob_root_config(
        "/home/User/code/stream-tuning/flink-tuner/config/full_params.yaml"
    )
    full_knobs = parse_knob_def(
        knob_selection_conf.knobs, conf.knobs.excluded_cluster_knob_prefixes
    )

    # ================= MODIFY BELOW =================
    log_file = "/home/User/code/stream-tuning/saved-results/20250626-132716"
    job_name = "q3"
    # ================= MODIFY ABOVE =================

    knob_names, df, weight_1, weight_2 = parse_selection_json(
        os.path.join(log_file, "observations.json")
    )

    plot_param_performance(
        df, knob_names, f"/home/User/Desktop/parameter_performance_{job_name}"
    )

    with pd.ExcelWriter(
        f"/home/User/Desktop/parameter_data_{job_name}.xlsx"
    ) as writer:
        df.to_excel(writer, index=False)

    importance_results = analyze_parameter_importance(knob_names, df)

    # Create an Excel writer object
    with pd.ExcelWriter(
        f"/home/User/Desktop/parameter_importance_{job_name}.xlsx"
    ) as writer:
        for metric, param_importances in importance_results.items():
            # Create a DataFrame from the parameter importances
            df = pd.DataFrame(
                list(param_importances.items()), columns=["Parameter", "Importance"]
            )

            # Sort the DataFrame by parameter names
            df = df.sort_values(by="Parameter")

            # Write the DataFrame to a sheet in the Excel file
            df.to_excel(writer, sheet_name=metric, index=False)

    # Select knobs
    selected_knobs = select_knobs(
        knob_names, df, importance_results["performance"], full_knobs
    )
