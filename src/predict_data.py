# Utils
import os
from flask import g
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Models and data processing
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

# Model utils
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from collections import defaultdict, Counter
try:
    from predict_utils import (
        get_ct_target,
        find_best_model,
        suggest_model,
        compute_medoid,
    )
except:
    from .predict_utils import (
        get_ct_target,
        find_best_model,
        suggest_model,
        compute_medoid,
    )


def run_experiment_2():
    """ Train an scikit-learn model on a CT-level classification task using
        cluster affordance of its eligibility criteria as input
    """
    for target_type in g.cfg["PREDICTOR_TARGET_TYPES"]:
        for input_type in g.cfg["PREDICTOR_INPUT_TYPES"]:
            
            # Take model and retrieve cluster result dir
            embed_model_id = g.cfg["PREDICTOR_EMBEDDING_MODEL_ID"]
            result_dir = os.path.join(g.cfg["RESULT_DIR"], embed_model_id)
            
            # Build dataset and create splits
            X, y = extract_prediction_dataset(result_dir, target_type, input_type)
            X_train, X_val_test, y_train, y_val_test = train_test_split(
                X, y,
                test_size=0.3, random_state=g.cfg["RANDOM_STATE"],
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test,
                test_size=0.5, random_state=g.cfg["RANDOM_STATE"],
            )
            if g.cfg["BALANCE_PREDICTION_DATA"]:
                smote = SMOTE(random_state=g.cfg["RANDOM_STATE"])
                X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Normalize input features and output targets
            normalizer_X = StandardScaler()
            normalizer_y = LabelEncoder()
            X_train = normalizer_X.fit_transform(X_train)
            X_val = normalizer_X.fit_transform(X_val)
            X_test = normalizer_X.transform(X_test)
            
            y_train = normalizer_y.fit_transform(y_train)
            y_val = normalizer_y.transform(y_val)
            y_test = normalizer_y.transform(y_test)
            
            # Find best hyper-parameters and fit final regression model
            g.logger.info("Finding best model for %s, %s" % (target_type, input_type))
            best_params = find_best_model(X_train, y_train, X_val, y_val)
            best_model = suggest_model(model_params=best_params)
            best_model.fit(X_train, y_train)
            
            # Test best model and write results to a directory
            file_name = "T%s-I%s-P%s" % (target_type, input_type, "-".join(g.cfg["CHOSEN_PHASES"]))
            output_path = os.path.join(result_dir, "predict_results", file_name)
            test_model(best_model, X_test, y_test, normalizer_y, output_path)
            
            
def extract_prediction_dataset(
    result_dir: str,
    target_type: str,
    input_type: str,
):
    """ Extract a dataset paring EC cluster affordances for any clinical trial
        to the corresponding target defined by target_type

    Args:
        result_dir (str): directory containing cluster results
        
    Returns:
        (tuple[np.ndarray, np.ndarray]): input feature and target arrays
    """
    # Collect raw embeddings and cluster output data
    raw_embedding_path = result_dir.replace("results/", "processed/embeddings_") + ".pt"
    raw_embeddings = torch.load(raw_embedding_path).numpy()
    cluster_result_path = os.path.join(result_dir, "ec_clustering.json")
    with open(cluster_result_path, "r") as f:
        cluster_output_data = json.load(f)
    
    # Pre-process each clinical trial data
    taken_ct_path_dict = {}
    cluster_dict = defaultdict(list)
    red_embedding_dict = defaultdict(list)
    raw_embedding_dict = defaultdict(list)
    raw_medoid_dict = defaultdict(list)
    red_medoid_dict = defaultdict(list)
    for cluster_instance in tqdm(
        iterable=cluster_output_data["cluster_instances"],
        total=len(cluster_output_data["cluster_instances"]),
        desc="Building input features from eligibility criteria",
        bar_format="{l_bar}{n_fmt}/{total_fmt} clusters{bar}{r_bar}",
    ):
        cluster_id = cluster_instance["cluster_id"]
        ec_list = cluster_instance["ec_list"]
        ec_ids = [ec["member_id"] for ec in ec_list]
        raw_cluster_medoid = compute_medoid(raw_embeddings[ec_ids])
        red_cluster_medoid = cluster_instance["medoid"]
        for ec, ec_id in zip(ec_list, ec_ids):
            # Check processed clinical trial algigns with chosen phase(s)
            ct_path = ec["ct_id"]
            if ct_path not in taken_ct_path_dict:
                with open(ct_path, "r", encoding="utf-8") as file:
                    ct_raw_dict: dict[str, dict|bool] = json.load(file)
                ct_phases = ct_raw_dict["protocolSection"]["designModule"]["phases"]
                ct_phases = [p.lower() for p in ct_phases]
                if all([p not in ct_phases for p in g.cfg["CHOSEN_PHASES"]])\
                and g.cfg["CHOSEN_PHASES"] != []:
                    taken_ct_path_dict[ct_path] = False
                else:
                    taken_ct_path_dict[ct_path] = True
            
            # Populate input feature data point
            if taken_ct_path_dict[ct_path]:
                raw_embedding = raw_embeddings[ec_id]
                red_embedding = ec["reduced_embedding"]
                cluster_dict[ct_path].append(cluster_id)
                red_embedding_dict[ct_path].append(red_embedding)
                raw_embedding_dict[ct_path].append(raw_embedding)
                raw_medoid_dict[ct_path].append(raw_cluster_medoid)
                red_medoid_dict[ct_path].append(red_cluster_medoid)
            
    # Initialize final dataset
    n_samples_max = len(cluster_dict)
    n_clusters = len(cluster_output_data["cluster_instances"])
    raw_embedding_dim = len(raw_embedding)  # last one loaded
    red_embedding_dim = len(red_embedding)  # last one loaded
    output_targets = np.empty((n_samples_max,), dtype=object)
    cluster_features = np.zeros((n_samples_max, n_clusters))
    raw_embedding_features = np.zeros((n_samples_max, raw_embedding_dim))
    red_embedding_features = np.zeros((n_samples_max, red_embedding_dim))
    raw_medoid_features = np.zeros((n_samples_max, raw_embedding_dim))
    red_medoid_features = np.zeros((n_samples_max, red_embedding_dim))
    
    # Extract feature target pairs
    for row_id, (
            (ct_path_1, cluster_data),
            (ct_path_2, raw_embedding_data),
            (ct_path_3, red_embedding_data),
            (ct_path_4, raw_medoid_data),
            (ct_path_5, red_medoid_data),
        ) in tqdm(
        iterable=enumerate(zip(
            cluster_dict.items(),
            raw_embedding_dict.items(),
            red_embedding_dict.items(),
            raw_medoid_dict.items(),
            red_medoid_dict.items(),
        )),
        total=len(cluster_dict),
        desc="Extracting feature-target pairs",
        bar_format="{l_bar}{n_fmt}/{total_fmt} CTs{bar}{r_bar}",
    ):
        # Load clinical trial data
        assert ct_path_1 == ct_path_2 == ct_path_3 == ct_path_4 == ct_path_5
        with open(ct_path_1, "r") as f:
            ct_data = json.load(f)
        
        # Extract target (or None if not found)
        ct_target = get_ct_target(ct_data, target_type)
        output_targets[row_id] = ct_target
        
        # Build cluster input feature vector from EC cluster affordances
        cluster_counts = Counter(cluster_data)
        cluster_col_ids = [n_clusters - 1 if c == -1 else c for c in cluster_counts.keys()]
        cluster_features[row_id][cluster_col_ids] = list(cluster_counts.values())
        
        # Build raw and reduced embedding feature vectors
        raw_embedding_features[row_id] = np.mean(raw_embedding_data, axis=0)
        red_embedding_features[row_id] = np.mean(red_embedding_data, axis=0)
        
        # Build raw and reduced medoid feature vectors
        raw_medoid_features[row_id] = np.mean(raw_medoid_data, axis=0)
        red_medoid_features[row_id] = np.mean(red_medoid_data, axis=0)
    
    # Build complete vectors that include both cluster and embedding information
    raw_complete_features = np.hstack((raw_embedding_features, cluster_features))
    red_complete_features = np.hstack((red_embedding_features, cluster_features))
    
    # Build final dataset (removing samples with no label found)
    mask = np.vectorize(lambda x: x is not None)(output_targets)
    output_targets = output_targets[mask]
    cluster_features = cluster_features[mask]
    raw_embedding_features = raw_embedding_features[mask]
    red_embedding_features = red_embedding_features[mask]
    raw_medoid_features = raw_medoid_features[mask]
    red_medoid_features = red_medoid_features[mask]
    raw_complete_features = raw_complete_features[mask]
    red_complete_features = red_complete_features[mask]
    
    # Select input feature type
    match input_type:
        case "cluster_ids":
            input_features = cluster_features
        case "raw_embeddings":
            input_features = raw_embedding_features
        case "reduced_embeddings":
            input_features = red_embedding_features
        case "raw_medoids":
            input_features = raw_medoid_features
        case "reduced_medoids":
            input_features = red_medoid_features
        case "raw_completes":
            input_features = raw_complete_features
        case "reduced_completes":
            input_features = red_complete_features
        case "random":
            input_features = np.random.rand(*cluster_features.shape)
        case _:
            raise ValueError("Invalid predictor input type")
    
    # Transform labels given predictor task type (regression vs classification)
    task_type = "classification" if target_type in ["phase", "status"] else "classification_pareto"
    match task_type:
        
        # Create binary classification task using the Pareto principle
        case "classification_pareto":
            assert all(isinstance(x, (int, float)) for x in output_targets)
            threshold = np.sort(output_targets)[int(len(output_targets) * 0.5)]  # 0.8?
            output_targets = np.where(output_targets >= threshold, 1, 0).astype("object")
        
        # Normal classification task
        case "classification":
            assert not any(isinstance(x, float) for x in output_targets)
            num_classes = len(list(set(output_targets)))
            print("Number of classes: %s" % num_classes)
        
    # Adapt label data type
    if output_targets.dtype == object:
        output_targets = output_targets.astype(type(output_targets[0]))
        
    return input_features, output_targets
    

def test_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalizer_y: Pipeline|StandardScaler,
    output_path: str,
    produce_img: bool=False,
):
    """ Test model, write classification report to a csv file and confusion
        matrix to an image file

    Args:
        output_path (str): where results are saved
        model (BaseEstimator): model trained with the training dataset
        X_test (np.ndarray): input features of the testing dataset
        y_test (np.ndarray): output targets of the testing dataset
        normalizer_y (Pipeline|StandardScaler): normalizer used during training
    """
    # Predict the test set using the best model
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values to compare them in their original scale
    y_test = normalizer_y.inverse_transform(y_test)
    y_pred = normalizer_y.inverse_transform(y_pred)
    
    # Write classification report to a csv file
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(output_path + ".csv")
    
    # Write confusion matrix to an image file
    if produce_img:
        cm = confusion_matrix(y_test, y_pred)
        class_names = normalizer_y.classes_
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        _, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(output_path + ".png")
        plt.close()
    
    # Log main performance indicator
    logged_perf = df_report["f1-score"]["macro avg"]
    g.logger.info("Test-macro-avg-f1-score for this model: %.04f" % logged_perf)
    

if __name__ == "__main__":
    run_experiment_2()
    