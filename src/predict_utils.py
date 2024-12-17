# Utils
import numpy as np
import dask.array as da
import optuna
from optuna.samplers import TPESampler, RandomSampler
from functools import partial
from datetime import datetime
from flask import g
from typing import Any
from cupyx.scipy.spatial.distance import cdist

# Models and data processing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def find_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """ Create an Optuna study and find the best hyperparameters

    Args:
        X_train / X_val (np.ndarray): input variables
        y_train / y_val (np.ndarray): output target values

    Returns:
        Scikit-learn model: trained model with best hyper-parameters
    """
    # Initialize objective and sampler
    objective = partial(
        optuna_objective_fn,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    if g.cfg["OPTUNA_SAMPLER"] == "tpe":
        sampler = TPESampler(seed=g.cfg["RANDOM_STATE"])
    else:
        sampler = RandomSampler(seed=g.cfg["RANDOM_STATE"])
    
    # Optimizer hyper-parameters
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(
        func=objective,
        n_trials=g.cfg["N_PREDICTION_OPTUNA_TRIALS"],
        show_progress_bar=True,
    )
    
    return study.best_params


def optuna_objective_fn(
    trial: optuna.trial.BaseTrial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """ Objective function to be optimized by optuna

    Args:
        trial (optuna.trial): current optuna trial
        X_train/X_val (np.ndarray): input variables
        y_train/y_val (np.ndarray): output target values
        
    Returns:
        float: value for the metric to optimize
    """
    model = suggest_model(trial)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_val, y_val, cv=5, scoring="f1_macro")
    
    return np.mean(scores)


def suggest_model(
    trial: optuna.trial.BaseTrial|None=None,
    model_params: dict|None=None,
):
    """ Suggest model hyper-parameters for one given trial

    Args:
        trial (optuna.trial.BaseTrial): current optuna trial
        model_params (dict): if not None, provide a model with given parameters

    Returns:
        scikit-learn model: model with suggested or provided hyper-parameters
    """
    assert not (trial is None and model_params is None)
    
    if model_params is None:
        model_type = trial.suggest_categorical(
                "model_type", [
                    "ridge", "lasso", "elasticnet",
                    # "sv", "knn",  <- does not improve performance (and takes long to run) 
                    # "decisiontree", "randomforest", "gradientboosting",  <- does not improve performance (and takes very long to run) 
                ]
            )
    else:
        model_type = model_params.pop("model_type")
        
    match model_type:
        
        case "ridge":
            model_class = RidgeClassifier
            if model_params is None:
                model_params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                }
                
        case "lasso":
            model_class = LogisticRegression
            if model_params is None:
                model_params = {
                    "penalty": "l1",
                    "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                    "C": trial.suggest_float("C", 0.01, 10.0),
                }
            else:
                model_params["penalty"] = "l1"
                
        case "elasticnet":
            model_class = LogisticRegression
            if model_params is None:
                model_params = {
                    "solver": "saga",
                    "penalty": trial.suggest_categorical("penalty", ["elasticnet"]),
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                }
            else:
                model_params["solver"] = "saga"

        case "sv":
            model_class = SVC
            if model_params is None:
                model_params = {
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                }

        case "knn":
            model_class = KNeighborsClassifier
            if model_params is None:
                model_params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                }
                
        case "decisiontree":
            model_class = DecisionTreeClassifier
            if model_params is None:
                model_params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }

        case "randomforest":
            model_class = RandomForestClassifier
            if model_params is None:
                model_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }

        case "gradientboosting":
            model_class = GradientBoostingClassifier
            if model_params is None:
                model_params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }
                
        case _:
            raise ValueError(f"Unsupported model: {model_type}")
        
    return model_class(**model_params)
    

class UnravelTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for reshaping
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.reshape(-1, 1)
    
    def inverse_transform(self, X):
        return X.reshape(-1)


class RavelTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for raveling
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.ravel()
    
    def inverse_transform(self, X):
        return X.reshape(-1, 1)


class LogScalingTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer for log scaling
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(1 + X)
    
    def inverse_transform(self, X):
        return np.expm1(X) - 1


def compute_medoid(data: np.ndarray) -> np.ndarray:
    """ Compute medoids of a subset of samples of shape (n_samples, n_features)
        Distance computations are made with dask to mitigate memory requirements
    """
    dask_data = da.from_array(data, chunks=1_000)
    def compute_distances(chunk): return cdist(chunk, data)
    distances = dask_data.map_blocks(compute_distances, dtype=float)
    sum_distances = distances.sum(axis=1).compute()
    medoid_index = sum_distances.argmin().get()
    return data[medoid_index]


def parse_date(date_str: str) -> datetime:
    """ Parse a string-formatted date time into a datetime object

    Args:
        date_str (str): string-formatted date time

    Returns:
        datetime: datetime object
    """
    for date_format in ["%Y-%m-%d", "%Y-%m", "%Y"]:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
        
    return None


def get_ct_target(ct_data: dict, target_type: str) -> Any:
    """ Extract target defined by target_type from clinical trial data

    Args:
        ct_data (dict): clinical trial data extracted from ct.gov json file
        target_type (str): the type of target to extract

    Returns:
        Any: extracted target
    """
    target = None
    protocol = ct_data.get("protocolSection", {})
    
    match target_type:
        
        case "phase":
            phases = protocol\
                .get("designModule", {})\
                .get("phases", {})
            if any(p in phases for p in ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]):
                target = phases[0]
                
        case "status":
            target = protocol\
                .get("statusModule", {})\
                .get("overallStatus", "")
        
        case "study_duration":
            start_date_str = protocol\
                .get("statusModule", {})\
                .get("startDateStruct", {})\
                .get("date", None)
            completion_date_str = protocol\
                .get("statusModule", {})\
                .get("completionDateStruct", {})\
                .get("date", None)
            
            if start_date_str and completion_date_str:
                start_date = parse_date(start_date_str)
                compl_date = parse_date(completion_date_str)
                study_duration_months = (compl_date - start_date).days / 30  # in months
                
                if study_duration_months > 0:
                    target = study_duration_months
                    
        case "enrollment_count":
            enrollment_count = protocol\
                .get("designModule", {})\
                .get("enrollmentInfo", {})\
                .get("count", None)
            if enrollment_count:
                target = int(enrollment_count)
        
        case "operational_rate":
            enrollment_count = protocol\
                .get("designModule", {})\
                .get("enrollmentInfo", {})\
                .get("count", None)
            start_date_str = protocol\
                .get("statusModule", {})\
                .get("startDateStruct", {})\
                .get("date", None)
            completion_date_str = protocol\
                .get("statusModule", {})\
                .get("completionDateStruct", {})\
                .get("date", None)
            
            if enrollment_count and start_date_str and completion_date_str:
                start_date = parse_date(start_date_str)
                compl_date = parse_date(completion_date_str)
                study_duration_months = (compl_date - start_date).days / 30  # in months
                
                if study_duration_months > 0:
                    target = int(enrollment_count) / study_duration_months
        
        case _:
            raise ValueError("Invalid label type")
    
    return target
