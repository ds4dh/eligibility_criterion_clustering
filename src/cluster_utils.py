# Config
import os
import sys
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")

# Utils
import re
import csv
import json
import random
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from itertools import product
from collections import defaultdict
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
logging.getLogger("distributed.utils_perf").setLevel(logging.ERROR)

# Optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler, RandomSampler
from cuml import HDBSCAN
from cuml.metrics.cluster import silhouette_score

# Evaluation
import torch
import dask.array as da
from sklearn.preprocessing import LabelEncoder
from cupyx.scipy.spatial.distance import cdist
from torchmetrics.clustering import DunnIndex
from sklearn.metrics import (
    davies_bouldin_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure as hcv_measure,
)

# Dimensionality reduction
from umap import UMAP
from cuml import UMAP as CUML_UMAP
from sklearn.decomposition import PCA
from cuml.decomposition import PCA as CUML_PCA
from sklearn.manifold import TSNE
from cuml.manifold import TSNE as CUML_TSNE
from cuml.common import logger as cuml_logger
from bertopic.dimensionality import BaseDimensionalityReduction


class ClusterGeneration:
    def __init__(self) -> None:
        """ Initialize clustering model based on HDBSCAN, but including
            subclustering and hyper-parameter tuning
        """
        cfg = config.get_config()  # load current configuration
        self.cluster_info = None
        self.n_cluster_max = cfg["N_CLUSTER_MAX"]
        self.optuna_param_ranges = cfg["OPTUNA_PARAM_RANGES"]
        self.n_optuna_trials = cfg["N_OPTUNA_TRIALS"]
        self.random_state = cfg["RANDOM_STATE"]
        if cfg["OPTUNA_SAMPLER"] == "tpe":
            self.sampler = TPESampler(seed=self.random_state)
        else:
            self.sampler = RandomSampler(seed=self.random_state)
                        
    def fit(self, X: np.ndarray) -> dict:
        """ Use optuna to determine best set of cluster parameters
        """
        # Initialize optuna study
        study = optuna.create_study(
            sampler=self.sampler,
            direction="maximize",
        )
        
        # Find best set of hyper-parameters
        logger_prefix = "[Training clustering algorithm with %s reduced eligibility criterion embeddings] " % len(X)
        logger_suffix = " [Optuna study optimizing cluster algorithm hyper-parameters]"
        objective = lambda trial: self.objective_fn(trial, X)
        with redirect_tqdm_to_logger(prefix=logger_prefix, suffix=logger_suffix):
            study.optimize(
                func=objective,
                n_trials=self.n_optuna_trials,
                show_progress_bar=True,
                gc_after_trial=True,
                n_jobs=1,
            )
        
        self.best_hyper_params = study.best_params
        self.labels_ = self.predict(X)
        return self
        
    def predict(self, X: np.ndarray) -> list[int]:
        """ Cluster samples and return cluster labels for each sample
        """
        params = self.best_hyper_params
        self.cluster_info = self.clusterize(data=X, params=params)
        # return self.cluster_info["clusterer"].labels_
        return self.cluster_info["cluster_ids"]

    def objective_fn(self, trial: optuna.Trial, data: np.ndarray) -> float:
        """ Suggest a new set of hyper-parameters and compute associated metric
        """
        # Perform clustering with a new set of suggested parameters
        params = self.suggest_parameters(trial)
        try:
            cluster_info = self.clusterize(data=data, params=params)
        except Exception as e:
            print(str(e))
            exit()
            return float("-inf")
        
        # Return metric from the clustering results
        if 1 < cluster_info["n_clusters"] < self.n_cluster_max:
            cluster_lbls = cluster_info["cluster_ids"]
            metric = 0.0
            metric += 1.0 * silhouette_score(data, cluster_lbls, chunksize=20_000)
            no_lbl_rate = np.count_nonzero(cluster_lbls == -1) / len(cluster_lbls)
            metric += 0.5 * (1.0 - no_lbl_rate)
            return metric
        else:
            return float("-inf")
    
    def suggest_parameters(self, trial: optuna.Trial) -> dict:
        """ Suggest parameters following configured parameter ranges and types
        """
        params = {}
        for name, choices in self.optuna_param_ranges.items():
            if not isinstance(choices, (list, tuple)):
                raise TypeError("Boundary must be a list or a tuple")
            if isinstance(choices[0], (str, bool)):
                params[name] = trial.suggest_categorical(name, choices)
                continue
            low, high = choices
            if isinstance(low, float) and isinstance(high, float):
                params[name] = trial.suggest_float(name, low, high)
            elif isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                raise TypeError("Boundary type mismatch")
        
        return params

    def set_cluster_params(self, n_samples: int, mode: str, params: dict) -> dict:
        """ Adapt clustering parameters following dataset size and clustering mode
        """
        # Load base parameters
        max_cluster_size = params["max_cluster_size_%s" % mode]
        min_cluster_size = params["min_cluster_size_%s" % mode]
        min_samples = params["min_samples_%s" % mode]
        cluster_selection_method = params["cluster_selection_method_%s" % mode]
        alpha = params["alpha_%s" % mode]
        
        # Adapter function (int vs float, min and max values)
        default_max_value = n_samples - 1
        def adapt_param_fn(value, min_value, max_value=default_max_value):
            if isinstance(value, float): value = int(n_samples * value)
            return min(max(min_value, value), max_value)
        
        # Return cluster parameters
        return {
            "cluster_selection_method": cluster_selection_method,
            "alpha": alpha,
            "allow_single_cluster": True,
            "max_cluster_size": adapt_param_fn(max_cluster_size, 100),
            "min_cluster_size": adapt_param_fn(min_cluster_size, 10),
            "min_samples": adapt_param_fn(min_samples, 10, 128),  # 128 # 1023
        }

    def clusterize(
        self,
        data: np.ndarray,
        params: dict,
        mode: str="primary",
    ) -> dict:
        """ Cluster data points with hdbscan algorithm and return cluster information
        """
        # Identify cluster parameters given the data and cluster mode
        cluster_params = self.set_cluster_params(len(data), mode, params)
        
        # Find cluster affordances based on cluster hierarchy
        set_seeds(self.random_state)  # try to ensure reproducibility
        clusterer = HDBSCAN(**cluster_params)
        clusterer.fit(data)
        
        # Identify main cluster ids
        cluster_ids = clusterer.labels_
        if mode == "secondary":
            cluster_ids[np.where(cluster_ids == -1)] = 0
        
        # Build lists of sample ids for each cluster
        n_clusters = np.max(cluster_ids).item() + 1  # -1 not counted in n_clusters
        member_ids = {
            k: np.where(cluster_ids == k)[0].tolist() for k in range(-1, n_clusters)
        }
        
        # Build cluster info
        cluster_info = {
            "clusterer": clusterer,
            "cluster_data": data,
            "n_clusters": n_clusters,
            "cluster_ids": cluster_ids,
            "member_ids": member_ids,
        }
        
        # Optionally sub-clusterize eligibility criteria embeddings
        if cluster_info["n_clusters"] > 1 and params["subclusterize"]:
            cluster_info = self.subclusterize(cluster_info, params=params)
        
        return cluster_info
     
    def subclusterize(self, cluster_info: dict, params: dict) -> dict:
        """ Update cluster results by trying to subcluster any computed cluster
        """
        # Rank cluster ids by cluster size
        cluster_lengths = {k: len(v) for k, v in cluster_info["member_ids"].items()}
        sorted_lengths = sorted(cluster_lengths.items(), key=lambda x: x[1], reverse=True)
        cluster_ranking = {k: rank for rank, (k, _) in enumerate(sorted_lengths, start=1)}
        threshold = int(np.ceil(cluster_info["n_clusters"] * 0.1))  # 10% largest clusters
        large_cluster_ids = [k for k, rank in cluster_ranking.items() if rank <= threshold]
        if -1 in large_cluster_ids: large_cluster_ids.remove(-1)
        
        # For large clusters, try to cluster it further with new hdbscan parameters
        for cluster_id in large_cluster_ids:
            subset_ids = np.where(cluster_info["cluster_ids"] == cluster_id)[0]
            subset_data = cluster_info["cluster_data"][subset_ids]
            new_cluster_info = \
                self.clusterize(data=subset_data, mode="secondary", params=params)
            
            # If the sub-clustering is successful, record new information
            if new_cluster_info["n_clusters"] > 1:  # new_cluster_info is not None:
                n_new_clusters = new_cluster_info["n_clusters"]
                new_cluster_ids =\
                    [cluster_id] +\
                    [cluster_info["n_clusters"] + i for i in range(n_new_clusters - 1)]
                cluster_info["n_clusters"] += n_new_clusters - 1
                
                # And update cluster labels and cluster member ids
                for i, new_cluster_id in enumerate(new_cluster_ids):
                    new_member_ids = new_cluster_info["member_ids"][i]
                    new_member_ids = subset_ids[new_member_ids]  # in original clustering
                    cluster_info["cluster_ids"][new_member_ids] = new_cluster_id
                    cluster_info["member_ids"][new_cluster_id] = new_member_ids
        
        # Return updated cluster info
        return cluster_info


@dataclass
class EligibilityCriterionData:
    ct_id: str
    member_id: int
    raw_text: str
    reduced_embedding: list[float]  # length = 2 (or plot_dim)


@dataclass
class ClusterInstance:
    cluster_id: int
    title: str
    n_samples: int
    prevalence: float
    medoid: list[float]  # length = 2 (or plot_dim)
    ec_list: list[EligibilityCriterionData]  # length = n_samples
    
    def __init__(
        self,
        cluster_id: int,
        ct_ids: list[str],
        member_ids: list[int],
        title: str,
        prevalence: float,
        medoid: np.ndarray,
        raw_txts: list[str],
        plot_data: np.ndarray,
    ) -> None:
        """ Build a cluster instance as requested by RisKlick
        """
        self.cluster_id = cluster_id
        self.title = title
        self.n_samples = len(raw_txts)
        self.prevalence = prevalence
        self.medoid = medoid
        self.ec_list = [
            EligibilityCriterionData(
                ct_id=ct_id,
                member_id=member_id,
                raw_text=raw_text,
                reduced_embedding=plot_embedding,
            )
            for ct_id, member_id, raw_text, plot_embedding,
            in zip(ct_ids, member_ids, raw_txts, plot_data.tolist())
        ]
        

@dataclass
class ClusterOutput:
    input_data_path: str
    output_dir: str
    user_id: str | None
    project_id: str | None
    embed_model_id: str
    cluster_metrics: dict[str, float]
    cluster_instances: list[ClusterInstance]
    visualization_paths: dict[str, dict[str, str]]
    raw_ec_list_path: str
    json_path: str
    
    def __init__(
        self,
        input_data_path: str,
        output_base_dir: str,
        user_id: str | None,
        project_id: str | None,
        embed_model_id: str,
        topic_model: BERTopic,
        raw_txts: list[str],
        metadatas: list[str],
        write_results: bool,
    ):
        """ Initialize a class to evaluate the clusters generated by an instance
            of ClusterGeneration, given a set of labels
        """
        # Load current configuration
        self.cfg = config.get_config()
        
        # Identify where results come from are where they are stored
        self.input_data_path = input_data_path
        self.embed_model_id = embed_model_id
        self.user_id = user_id
        self.project_id = project_id
        if write_results:
            self.output_dir = os.path.join(output_base_dir, embed_model_id)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = None
        
        # Raw data and labels for each eligibility criterion
        self.raw_txts = raw_txts
        self.n_samples = len(raw_txts)
        self.ct_paths = [m["path"][0] for m in metadatas]
        self.phases = [l["phase"] for l in metadatas]
        self.conds = [l["condition"] for l in metadatas]
        self.itrvs = [l["intervention"] for l in metadatas]
        
        # Cluster data and evaluation
        logger.info("Formatting cluster data")
        self.cluster_info = self.get_cluster_info(topic_model)
        logger.info("Generating cluster titles from representation model")
        self.cluster_titles = self.get_cluster_titles(topic_model)
        logger.info("Reducing cluster data dimensionality for plotting")
        self.plot_data = self.get_plot_data()
        if self.cfg["DO_EVALUATE_CLUSTERING"]:
            logger.info("Evaluating cluster quality")
            self.cluster_metrics = self.evaluate_clustering()
        else:
            self.cluster_metrics = None
        logger.info("Computing statistics for each cluster")
        self.statistics = self.compute_cluster_statistics()
        logger.info("Retrieving individual cluster data")
        self.cluster_instances = self.get_cluster_instances()
        
        # Cluster visualization and reports
        if write_results:
            logger.info("Plotting clusters")
            self.visualization_paths = self.plot_clusters()
            logger.info("Writing raw cluster lists")
            self.raw_ec_list_path = self.write_raw_ec_list()
            logger.info("Writing formatted cluster json file")
            self.write_to_json()  # this sets self.json_path
            logger.info("Cluster results have been generated!")
        else:
            self.visualization_paths = None
            self.raw_ec_list_path = None
            self.json_path = None
    
    def get_cluster_info(self, topic_model: BERTopic) -> dict:
        """ Re-align cluster ids from ClusterGeneration object to BERTopic topics
            BERTopic sorts topics, and hence disaligns with original cluster ids
        """
        original_cluster_info = topic_model.hdbscan_model.cluster_info
        cluster_ids = np.array(topic_model.topics_)
        min_cluster_id = cluster_ids.min()  # -1 may or may not be in cluster_ids
        n_clusters = cluster_ids.max() + 1  # -1 not counted in n_cluster
        member_ids = {
            k: np.where(cluster_ids == k)[0].tolist()
            for k in range(min_cluster_id, n_clusters)
        }
        return {
            "clusterer": original_cluster_info["clusterer"],
            "cluster_data": original_cluster_info["cluster_data"],
            "n_clusters": n_clusters,
            "min_cluster_id": min_cluster_id,  # -1 or 0
            "cluster_ids": cluster_ids,
            "member_ids": member_ids,
        }
    
    def get_cluster_titles(self, topic_model: BERTopic) -> dict:
        """ Format titles from raw BERTopic representation model output
        """
        # Helper function that adapts to diverse representation formats
        def get_title(representation) -> str:
            """ Find cluster title from its bertopic representation
            """
            if isinstance(representation, str):
                return representation
            elif isinstance(representation, (tuple, list)):
                titles = [get_title(r) for r in representation]
                return "-".join([t for t in titles if t])
        
        # Return formatted BERTopic representatinos
        formatted_titles = {
            k: get_title(v) if k != -1 else "Sample with unidentified cluster"
            for k, v in topic_model.topic_representations_.items()
        }
        return formatted_titles
    
    def get_cluster_instances(self) -> list[ClusterInstance]:
        """ Separate data by cluster and build a formatted cluster instance for each
        """
        cluster_instances = []
        for cluster_id, member_ids in self.statistics["sorted_member_ids"].items():
            
            cluster_instances.append(
                ClusterInstance(
                    cluster_id=cluster_id,
                    ct_ids=[self.ct_paths[i] for i in member_ids],
                    member_ids=member_ids,
                    title=self.cluster_titles[cluster_id],
                    prevalence=self.statistics["prevalences"][cluster_id],
                    medoid=self.statistics["medoids"][cluster_id].tolist(),
                    raw_txts=[self.raw_txts[i] for i in member_ids],
                    plot_data=self.plot_data[member_ids],
                )
            )
        
        # Function sorting clusters by number of samples, and with "-1" last
        def custom_sort_key(cluster_instance: ClusterInstance, sort_by: str="size"):
            if cluster_instance.cluster_id == -1:
                return 0  # to go to the last position
            if sort_by == "size":
                return cluster_instance.n_samples
            elif sort_by == "prevalence":
                return cluster_instance.prevalence
        
        # Return cluster instances (sorting helps printing data structure)
        return sorted(cluster_instances, key=custom_sort_key, reverse=True)
    
    def get_plot_data(self) -> np.ndarray:
        """ Compute low-dimensional plot data using t-SNE algorithm, or take it
            directly from the cluster data if it has the right dimension
        """
        # if self.cfg["CLUSTER_RED_DIM"] == self.cfg["PLOT_RED_DIM"] \
        # and self.cfg["CLUSTER_DIM_RED_ALGO"] == self.cfg["PLOT_DIM_RED_ALGO"] \
        # or self.cfg["CLUSTER_DIM_RED_ALGO"] is None:
        if self.cfg["CLUSTER_RED_DIM"] == self.cfg["PLOT_RED_DIM"] \
        or self.cfg["CLUSTER_DIM_RED_ALGO"] is None:
            return self.cluster_info["cluster_data"]
        else:
            dim_red_model = get_dim_red_model(
                self.cfg["PLOT_DIM_RED_ALGO"],
                self.cfg["PLOT_RED_DIM"],
                self.n_samples,
            )
            return dim_red_model.fit_transform(self.cluster_info["cluster_data"])
    
    def compute_cluster_statistics(self) -> dict:
        """ Compute CT prevalence between clusters & label prevalence within clusters
        """
        # Match clinical trial ids to cluster ids for all criteria
        cluster_ids = self.cluster_info["cluster_ids"].tolist()
        zipped_paths = list(zip(self.ct_paths, cluster_ids))
        
        # Compute absolute cluster prevalence by counting clinical trials
        n_cts = len(set(self.ct_paths))
        cluster_sample_paths = {
            cluster_id: [p for p, l in zipped_paths if l == cluster_id]
            for cluster_id in range(
                self.cluster_info["min_cluster_id"],  # -1 or 0
                self.cluster_info["n_clusters"],
            )
        }
        cluster_prevalences = {
            cluster_id: len(set(paths)) / n_cts  # len(paths) / token_info["ct_ids"]
            for cluster_id, paths in cluster_sample_paths.items()
        }
        
        # Sort cluster member ids by how close each member is to its cluster medoid
        cluster_medoids = self.compute_cluster_medoids(self.cluster_info)
        cluster_data = self.cluster_info["cluster_data"]
        cluster_member_ids = self.cluster_info["member_ids"]
        cluster_sorted_member_ids = {}
        for k, member_ids in cluster_member_ids.items():
            medoid = cluster_medoids[k][np.newaxis, :]
            members_data = cluster_data[member_ids]
            distances = cdist(medoid, members_data)[0]
            sorted_indices = np.argsort(np.nan_to_num(distances).flatten())
            cluster_sorted_member_ids[k] = [
                member_ids[idx] for idx in sorted_indices.get()
            ]
        
        return {
            "prevalences": cluster_prevalences,
            "medoids": cluster_medoids,
            "sorted_member_ids": cluster_sorted_member_ids,
        }
    
    def compute_cluster_medoids(self, cluster_info: dict) -> dict:
        """ Compute cluster centroids by averaging samples within each cluster,
            weighting by the sample probability of being in that cluster
        """
        cluster_medoids = {}
        for label in range(
            self.cluster_info["min_cluster_id"],  # -1 or 0
            cluster_info["n_clusters"],
        ):
            cluster_ids = np.where(cluster_info["cluster_ids"] == label)[0]
            cluster_data = cluster_info["cluster_data"][cluster_ids]
            cluster_medoids[label] = self.compute_medoid(cluster_data)
            
        return cluster_medoids
    
    @staticmethod
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
    
    def evaluate_clustering(self):
        """ Run final evaluation of clusters, based on phase(s), condition(s), and
            interventions(s). Duplicate each samples for any combination.    
        """
        # Get relevant data
        cluster_metrics = {}
        cluster_data = self.cluster_info["cluster_data"]
        cluster_lbls = self.cluster_info["cluster_ids"]
        
        # Evaluate clustering quality (label-free)
        sil_score = silhouette_score(cluster_data, cluster_lbls, chunksize=20_000)
        cluster_metrics["label_free"] = {
            "Silhouette score": sil_score,
            "DB index": davies_bouldin_score(cluster_data, cluster_lbls),
            "Dunn index": self.dunn_index(cluster_data, cluster_lbls),
        }
        
        # Fill-in true and cluster ("dept") labels, duplicating samples
        # when the corresponding clinical trial has multiple combinations
        # of phase/condition/intervention
        cluster_lbls = cluster_lbls.tolist()
        dupl_lbls = defaultdict(list)
        for cluster_lbl, phases, conds, itrvs, ct_path in\
            zip(cluster_lbls, self.phases, self.conds, self.itrvs, self.ct_paths):
            if cluster_lbl != -1:
                
                # One label = one combination of phase/condition/intervention
                true_lbl_combinations = list(product(phases, conds, itrvs))
                for i, true_lbl_combination in enumerate(true_lbl_combinations):
                    true_lbl = "- ".join(true_lbl_combination)
                    dupl_lbls["true"].append(true_lbl)
                    dupl_lbls["dept"].append(cluster_lbl)
                    
        #########################################################################
        # ADDED FOR REVISION 1
        
        # Build random and optimal inferred labels to compute floor and ceiling
        # performance by assigning a sample to the true label but assuming other
        # samples from the same clinical trial are in different clusters
        all_true_labels = list(set(dupl_lbls["true"]))
        used_ids = set()
        for cluster_lbl, phases, conds, itrvs, ct_path in\
            zip(cluster_lbls, self.phases, self.conds, self.itrvs, self.ct_paths):
            if cluster_lbl != -1:
                
                # One label = one combination of phase/condition/intervention
                true_lbl_combinations = list(product(phases, conds, itrvs))
                for i, true_lbl_combination in enumerate(true_lbl_combinations):
                    used_id = f"{ct_path}_{i}"  # accounting for duplicated samples
                    optim_lbl = true_lbl if used_id not in used_ids else "other"
                    used_ids.add(used_id)
                    dupl_lbls["ceil"].append(f"{optim_lbl}_{i}")
                    dupl_lbls["rand"].append(random.choice(all_true_labels))
                    
        # ADDED FOR REVISION 1
        #########################################################################
        
        # Evaluate clustering quality based on true labels
        true_encoder = LabelEncoder()
        true_lbls = true_encoder.fit_transform(dupl_lbls["true"]).astype(np.int32)
        for pred_type in ["dept", "rand", "ceil"]:
            
            # Compute label-dependent metrics, comparing inferred labels to true labels
            pred_encoder = LabelEncoder()
            pred_lbls = pred_encoder.fit_transform(dupl_lbls[pred_type]).astype(np.int32)
            homogeneity, completeness, v_measure = hcv_measure(true_lbls, pred_lbls)
            cluster_metrics["label_%s" % pred_type] = {
                "Homogeneity": homogeneity,
                "Completeness": completeness,
                "V measure": v_measure,
                "MI score": mutual_info_score(true_lbls, pred_lbls),
                "AMI score": adjusted_mutual_info_score(true_lbls, pred_lbls),
                "AR score": adjusted_rand_score(true_lbls, pred_lbls),
            }
        
        # Update cluster_info main dict with cluster_metrics
        cluster_metrics["n_samples"] = len(cluster_lbls)
        cluster_metrics["n_duplicated_samples"] = len(dupl_lbls["dept"])
        
        return cluster_metrics
    
    def build_subset_ct_to_cluster_matrix(
        self,
        subset_key: str,  # specific condition or intervention
        cond_or_itrv: str,  # "cond", "itrv"
        pos_or_neg: str,  # "pos", "neg"
    ):
        """ Build a matrix that summarizes cluster affordances for every clinical
            trial that includes a given condition or intervention
        """
        assert cond_or_itrv in ["cond", "itrv"]
        if cond_or_itrv == "cond":
            keys = self.conds
        else:
            keys = self.itrvs
            
        assert pos_or_neg in ["pos", "neg"]
        if pos_or_neg == "pos":
            subset_ec_ids = [i for i, cs in enumerate(keys) if subset_key in cs]
        else:
            subset_ec_ids = [i for i, cs in enumerate(keys) if subset_key not in cs]
            
        cluster_lbls = self.cluster_info["cluster_ids"]
        subset_ct_paths = [self.ct_paths[i] for i in subset_ec_ids]
        subset_cluster_lbls = [cluster_lbls[i] for i in subset_ec_ids]
        # subset_cluster_lbls = [l for l in subset_cluster_lbls if l != -1]
        
        unique_subset_ct_paths = list(set(subset_ct_paths))
        unique_cluster_labels = list(set(cluster_lbls))
        n_subset_ct_paths = len(unique_subset_ct_paths)
        n_cluster_lbls = len(unique_cluster_labels)
        
        if len(unique_subset_ct_paths) <= 10:
            return None
        
        ct_path_keys = {ct_id: i for i, ct_id in enumerate(unique_subset_ct_paths)}
        ct_to_cluster_matrix = np.zeros((n_cluster_lbls, n_subset_ct_paths), dtype=int)
        for cluster_lbl, ct_id in zip(subset_cluster_lbls, subset_ct_paths):
            ct_to_cluster_matrix[cluster_lbl, ct_path_keys[ct_id]] = 1
        
        return ct_to_cluster_matrix
    
    @staticmethod
    def dunn_index(cluster_data: np.ndarray, cluster_lbls: np.ndarray) -> float:
        """ Compute Dunn index using torch-metrics
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dunn = DunnIndex(p=2)
        metric = dunn(
            torch.as_tensor(cluster_data, device=device),
            torch.as_tensor(cluster_lbls, device=device),
        )
        return metric.item()
    
    def plot_clusters(self, font_size: float=21.0, top_k: int=20) -> None:
        """ Plot clusters as a coloured scatter plot, both for all cluster,
            (without labels) and for the top_k largest clusters (with labels)
        """
        # Initialize variables
        assert top_k <= 20, "top_k must not be greater than 20"
        visualization_paths = defaultdict(lambda: {})
        is_3d = len(self.cluster_instances[0].ec_list[0].reduced_embedding) == 3
        
        # Generate a visualization for top_k clusters and all clusters
        for do_top_k in [True, False]:
            
            # Retrieve cluster data (clusters are already sorted by n_sample)
            clusters = [c for c in self.cluster_instances]  # if c.cluster_id != -1]
            if do_top_k:
                clusters = clusters[:top_k]
                symbol_seq = ["circle", "square", "diamond", "x"]
                size_seq = [1.0, 0.7, 0.7, 0.5] if is_3d else [1.0, 1.0, 1.0, 1.0]
            else:
                symbol_seq = ["circle"]
                size_seq = [1.0]
            symbol_map = {k: symbol_seq[k % len(symbol_seq)] for k in range(100)}
            size_map = {k: size_seq[k % len(size_seq)] for k in range(100)}
            plotly_colors = px.colors.qualitative.Plotly
            
            # Format data for dataframe
            ys, xs, zs, raw_texts = [], [], [], []
            labels, hover_names, ids, symbols, sizes = [], [], [], [], []
            color_map = {}
            legend_line_count = 0
            for k, cluster in enumerate(clusters):
                
                # Cluster data
                label, label_line_count = self.format_text(
                    cluster.title, max_length=70, max_line_count=2,
                )
                legend_line_count += label_line_count
                labels.extend([label] * len(cluster.ec_list))
                ids.extend([cluster.cluster_id] * len(cluster.ec_list))
                hover_names.extend([self.format_text(
                    cluster.title, max_length=40, max_line_count=10,
                )[0]] * len(cluster.ec_list))
                
                # Eligibility criteria data
                xs.extend([ec.reduced_embedding[0] for ec in cluster.ec_list])
                ys.extend([ec.reduced_embedding[1] for ec in cluster.ec_list])
                if is_3d:
                    zs.extend([ec.reduced_embedding[2] for ec in cluster.ec_list])
                raw_texts.extend([self.format_text(
                    ec.raw_text, max_length=35, max_line_count=10,
                )[0] for ec in cluster.ec_list])
                
                # Eligibility criteria markers
                symbol = k // 10 if cluster.cluster_id != -1 else 0
                symbols.extend([symbol] * len(cluster.ec_list))
                size = size_map[k // 10] if cluster.cluster_id != -1 else 0.1
                sizes.extend([size] * len(cluster.ec_list))
                color = plotly_colors[k % 10] if cluster.cluster_id != -1 else "white"
                color_map[label] = color
                
            # Build dataframe for plotly.scatter
            plot_df = pd.DataFrame({
                "x": xs, "y": ys, "raw_text": raw_texts, "label": labels,
                "id": ids,  "hover_name": hover_names, "symbol": symbols,
                "size": sizes,
            })
            if is_3d: plot_df["z"] = zs
            
            # Plot cluster data using px.scatter
            hover_data = {
                "label": False, "hover_name": False, "raw_text": True,
                "symbol": False, "size": False, "x": ":.2f", "y": ":.2f",
            }
            if not is_3d:  # i.e., is_2d
                fig = px.scatter(
                    plot_df, x="x", y="y", opacity=1.0,
                    color="label", color_discrete_map=color_map,
                    labels={"label": "Cluster labels"}, size="size",
                    symbol="symbol", symbol_map=symbol_map,
                    hover_name="hover_name", hover_data=hover_data,
                )
            else:
                hover_data.update({"z": ":.2f"})
                fig = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", opacity=1.0,
                    color="label", color_discrete_map=color_map,
                    labels={"label": "Cluster labels"}, size="size",
                    symbol="symbol", symbol_map=symbol_map,
                    hover_name="hover_name", hover_data=hover_data,
                )
            
            # Polish figure
            legend_font_size = max(1, font_size * 20 / legend_line_count)
            legend_font_size = min(font_size, legend_font_size)  # not bigger
            width = 375
            height = width / 1.25
            if do_top_k:
                added_width = width / 70 * min(70, max([len(l) for l in labels]))
                added_width *= legend_font_size / font_size
                width += added_width
                marker_dict = dict(line=dict(color="black", width=1.0))
            else:
                marker_dict = dict(opacity=0.5, line=dict(color="gray", width=0.5))
            fig.update_traces(marker=marker_dict)
            fig.update_xaxes(
                # title_text="tSNE-1", linecolor="black", linewidth=0.5,
                # title_font=dict(size=font_size, family="TeX Gyre Pagella"),
                # tickfont=dict(size=font_size, family="TeX Gyre Pagella"),
                visible=False,
            )
            fig.update_yaxes(
                # title_text="tSNE-2", linecolor="black", linewidth=0.5,
                # title_font=dict(size=font_size, family="TeX Gyre Pagella"),
                # tickfont=dict(size=font_size, family="TeX Gyre Pagella"),
                visible=False,
            )
            fig.update_layout(
                plot_bgcolor="white", dragmode="pan",
                width=width, height=height,
                legend=dict(
                    yanchor="middle", y=0.5, xanchor="left",
                    title_text="", itemsizing="constant",
                    font=dict(size=legend_font_size, family="TeX Gyre Pagella"),
                )
            )
            if do_top_k:
                for trace in fig.data:
                    trace.name = trace.name.replace(', 0', '').replace(', 1', '')
            else:
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                )
            ms_factor = 1.0
            if not is_3d: ms_factor *= 0.3
            if not do_top_k: ms_factor *= 0.3                
            for trace in fig.data:
                if 'size' in trace.marker:
                    trace.marker.size = [s * ms_factor for s in trace.marker.size]
                    
            # Save image and sets plot_path
            plot_tag = "top_%i" % top_k if do_top_k else "all"
            plot_name = "cluster_plot_%s.png" % plot_tag
            plot_path = os.path.join(self.output_dir, plot_name)
            html_path = plot_path.replace("png", "html")
            fig.write_image(plot_path, engine="kaleido", scale=2)
            fig.write_html(html_path)
            visualization_paths[plot_tag]["png"] = plot_path
            visualization_paths[plot_tag]["html"] = html_path
        
        return dict(visualization_paths)
    
    @staticmethod
    def format_text(
        text: str,
        max_length: int=70,
        max_line_count: int=2
    ) -> tuple[str, int]:
        """ Format text by cutting it into lines and trimming it if too long
        """
        # Shorten criterion type information
        text = text.replace("\n", " ").replace("<br>", " ") \
            .replace("Inclusion criterion", "IN").replace("Inclusion", "IN") \
            .replace("inclusion criterion", "IN").replace("inclusion", "IN") \
            .replace("Exclusion criterion", "EX").replace("Exclusion", "EX") \
            .replace("exclusion criterion", "EX").replace("exclusion", "EX")
        
        # Let the text as it is if its length is ok
        if len(text) <= max_length:
            return text, 1
        
        # Split the title into words and build the formatted text
        words = re.findall(r'\w+-|\w+', text)
        shortened_text = ""
        current_line_length = 0
        line_count = 1
        for word in words:
            
            # Check if adding the next word would exceed the maximum length
            added_space = (" " if word[-1] != "-" else "")
            if current_line_length + len(word) > max_length:
                if line_count == max_line_count:  # replace remaining text by "..."
                    shortened_text = shortened_text.rstrip() + "..."
                    break
                else:  # Move to the next line
                    shortened_text += "<br>" + word + added_space
                    current_line_length = len(word) + len(added_space)
                    line_count += 1
            else:
                shortened_text += word + added_space
                current_line_length += len(word) + len(added_space)
        
        return shortened_text.strip(), line_count
    
    def write_raw_ec_list(self) -> str:
        """ Generate a raw list of criteria grouped by cluster
        """
        # Open the CSV file in write mode
        raw_ec_list_path = os.path.join(self.output_dir, "raw_ec_list.csv")
        with open(raw_ec_list_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                "Cluster id", "Cluster prevalence",
                "Cluster Title", "Eligibility Criteria",
            ])
            
            # Iterate through ClusterOutput objects and their ClusterInstances
            for cluster_instance in self.cluster_instances:
                title = cluster_instance.title
                cluster_id = cluster_instance.cluster_id
                prevalence = cluster_instance.prevalence
                for ec_data in cluster_instance.ec_list:
                    ec_text = ec_data.raw_text
                    csv_writer.writerow([cluster_id, prevalence, title, ec_text])
        
        return raw_ec_list_path
    
    def write_to_json(self) -> str:
        """ Convert cluster output to a dictionary and write it to a json file,
            after generating a unique file name given by the project and user ids
        """
        # Define file name and sets json_path
        json_path = os.path.join(self.output_dir, "ec_clustering.json")
        self.json_path = json_path  # need to set it here for asdict(self)
        
        # Save data as a json file
        cluster_output_dict = asdict(self)
        json_data = json.dumps(cluster_output_dict, indent=4)
        with open(json_path, "w") as file:
            file.write(json_data)
        
        return json_path
    

class CUML_TSNEForBERTopic(CUML_TSNE):
    def transform(self, X):
        logger.info("Reducing %s eligibility criterion embedding dimensionality" % len(X))
        reduced_X = self.fit_transform(X)
        return reduced_X.to_host_array()


class TSNEForBERTopic(TSNE):
    def transform(self, X):
        logger.info("Reducing %s eligibility criterion embedding dimensionality" % len(X))
        reduced_X = self.fit_transform(X)
        return reduced_X


def set_seeds(seed_value: int=1234):
    """ Set seed for reproducibility
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If using PyTorch and you want determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dim_red_model(algorithm: str, dim: int, n_samples: int):
    """ Create a dimensionality reduction model for BERTopic
    """
    # Load current configuration
    cfg = config.get_config()
    
    # No dimensionality reduction
    if algorithm is None:
        return BaseDimensionalityReduction()
    
    # Uniform manifold approximation and projection
    elif algorithm == "umap":
        n_neighbors = min(int(n_samples / 400 + 1), 90)
        return CUML_UMAP(
            n_components=dim,
            random_state=cfg["RANDOM_STATE"],
            n_neighbors=90,  # same as for t-SNE (see below) - UMAP default = 15
            learning_rate=200.0,  # same as for t-SNE (see below) - UMAP default = 1.0
            min_dist=0.0,  # same as for t-SNE (see below) - UMAP default = 0.1
            metric="correlation",  # same as for t-SNE (see below) - UMAP default = "euclidean"
        )
    
    # Principal component analysis
    elif algorithm == "pca":
        return CUML_PCA(
            n_components=dim,
            random_state=cfg["RANDOM_STATE"],
        )
    
    # t-distributed stochastic neighbor embedding
    elif algorithm == "tsne":
        params = {
            "n_components": dim,
            "random_state": cfg["RANDOM_STATE"],
            "method": "barnes_hut" if dim < 4 else "exact",  # "fft" or "barnes_hut"?
            "n_iter": cfg["N_ITER_MAX_TSNE"],
            "n_iter_without_progress": 1000,
            "metric": "correlation",
            "learning_rate": 200.0,
            "perplexity": 50.0,  # CannyLabs CUDA-TSNE default is 50
        }
        if n_samples < 36_000 and dim == 2:
            n_neighbors = min(int(n_samples / 400 + 1), 90)
            small_cuml_specific_params = {
                "n_neighbors": n_neighbors,  # CannyLabs CUDA-TSNE default is 32
                "learning_rate_method": "none",  # not in sklearn and produces bad results
            }
            params.update(small_cuml_specific_params)
        if dim == 2:
            params.update({"verbose": cuml_logger.level_error})
            return CUML_TSNEForBERTopic(**params)
        else:
            params.update({"verbose": False})
            return TSNEForBERTopic(**params)  # cuml_tsne only available for dim = 2
    
    # Invalid name
    else:
        raise ValueError("Invalid name for dimensionality reduction algorithm")


@contextmanager
def redirect_tqdm_to_logger(prefix: str="", suffix: str=""):
    class TqdmToLogger:
        def __init__(self, logger: config.CTxAILogger, prefix: str, suffix: str):
            self.logger = logger
            self.prefix = prefix
            self.suffix = suffix
            self.buf = ""
            
        def write(self, message: str):
            # Filter out empty or redundant messages
            if message.strip():
                self.logger.info(f"{self.prefix}{message.strip()}{self.suffix}")
                
        def flush(self):
            pass  # no-op for compatibility
    
    # Redirect tqdm output to global logger
    original_stderr = sys.stderr
    sys.stderr = TqdmToLogger(logger, prefix, suffix)
    try:
        yield
    finally:
        sys.stderr = original_stderr  # restore the original stderr