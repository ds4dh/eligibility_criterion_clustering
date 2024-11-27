# Config
import os
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Utils
import pandas as pd
import matplotlib.pyplot as plt
import torch
try:
    from cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model, set_seeds
    from parse_utils import get_embeddings
except:
    from .cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model, set_seeds
    from .parse_utils import get_embeddings

# Clustering and representation
from openai import OpenAI as OpenAIClient
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import CountVectorizer


def run_experiment_1():
    """ If ran as a script, call cluster_data for several models and write
        a summary of results to the output directory
    """
    # Load current configuration
    cfg = config.get_config()
    
    # Generate results using one model, then save the results
    cluster_metrics = {}
    for embed_model_id in cfg["EMBEDDING_MODEL_ID_MAP"].keys():
        logger.info("Starting with %s" % embed_model_id)
        cluster_output = cluster_data_fn(embed_model_id=embed_model_id)
        cluster_metrics[embed_model_id] = cluster_output.cluster_metrics
        logger.info("Done with %s" % embed_model_id)
    
    # Plot final results (comparison of different model embeddings)
    output_path = os.path.join(cfg["RESULT_DIR"], "model-comparison.png")
    if cfg["DO_EVALUATE_CLUSTERING"]:
        fig_title = "Label-dependent metrics %s" % cfg["CHOSEN_COND_IDS"]
        plot_model_comparison(cluster_metrics, output_path, fig_title)
    logger.info("Model comparison finished!")


def cluster_data_fn(
    embed_model_id: str,
    write_results: bool=True,
    hierarchical_ec_scraping: bool=False,
) -> ClusterOutput:
    """ Cluster eligibility criteria using embeddings from one language model
    """
    # Initialization
    cfg = config.get_config()
    if write_results:
        os.makedirs(cfg["PROCESSED_DIR"], exist_ok=True)
        os.makedirs(cfg["RESULT_DIR"], exist_ok=True)
    set_seeds(cfg["RANDOM_STATE"])  # try to ensure reproducibility
    bertopic_ckpt_path = os.path.join(
        cfg["PROCESSED_DIR"],
        "bertopic_model_%s" % embed_model_id,
    )
    
    # Generate or load elibibility criterion texts, embeddings, and metadatas
    logger.info("Getting elibility criteria embeddings from %s" % embed_model_id)
    embeddings, raw_txts, metadatas = get_embeddings(
        embed_model_id=embed_model_id,
        preprocessed_dir=cfg["PREPROCESSED_DIR"],
        processed_dir=cfg["PROCESSED_DIR"],
        hierarchical_ec_scraping=hierarchical_ec_scraping,
    )
        
    # Generate cluster representation with BERTopic
    if not cfg["LOAD_BERTOPIC_RESULTS"]:
        topic_model = train_bertopic_model(raw_txts, embeddings)
        if cfg["ENVIRONMENT"] == "ctgov"\
        and cfg["CLUSTER_REPRESENTATION_MODEL"] is None\
        and write_results:
            topic_model.save(bertopic_ckpt_path)
    
    # Load BERTopic cluster representation from previous run (only for ctgov)
    else:
        logger.info("Loading BERTopic model trained on eligibility criteria embeddings")
        topic_model = BERTopic.load(bertopic_ckpt_path)
        if cfg["REGENERATE_REPRESENTATIONS_AFTER_LOADING_BERTOPIC_RESULTS"]:
            topic_model.update_topics(
                docs=raw_txts,
                representation_model=get_representation_model(),
            )
    
    # Generate results from the trained model and predictions
    logger.info("Writing clustering results with bertopic titles")
    return ClusterOutput(
        input_data_path=cfg["FULL_DATA_PATH"],
        output_base_dir=cfg["RESULT_DIR"],
        user_id=cfg["USER_ID"],
        project_id=cfg["PROJECT_ID"],
        embed_model_id=embed_model_id,
        topic_model=topic_model,
        raw_txts=raw_txts,
        metadatas=metadatas,
        write_results=write_results,
    )


def train_bertopic_model(
    raw_txts: list[str],
    embeddings: torch.Tensor,
):
    """ Train a BERTopic model
    """
    # Create BERTopic model
    cfg = config.get_config()
    topic_model = BERTopic(
        top_n_words=cfg["CLUSTER_REPRESENTATION_TOP_N_WORDS_PER_TOPIC"],
        umap_model=get_dim_red_model(
            cfg["CLUSTER_DIM_RED_ALGO"],
            cfg["CLUSTER_RED_DIM"],
            len(embeddings),
        ),
        hdbscan_model=ClusterGeneration(),
        vectorizer_model=CountVectorizer(stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(),
        representation_model=get_representation_model(),
        verbose=True,
    )
    
    # Train BERTopic model using raw text documents and pre-computed embeddings
    logger.info(f"Running bertopic algorithm on {len(raw_txts)} embeddings")
    topic_model = topic_model.fit(raw_txts, embeddings)
    # topics = topic_model.reduce_outliers(raw_txts, topics)
    return topic_model


def get_representation_model():
    """ Get a model to represent each cluster-topic with a title
    """
    # Get current configuration
    cfg = config.get_config()
    
    # Prompt chat-gpt with keywords and document content
    if cfg["CLUSTER_REPRESENTATION_MODEL"] == "gpt":
        api_path = os.path.join(cfg["CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY"])
        try:
            with open(api_path, "r") as f: api_key = f.read()
        except:
            raise FileNotFoundError(" ".join([
                "To use CLUSTER_REPRESENTATION_MODEL = gpt,",
                "you must have an api-key file at the path defined in the",
                "config under CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY",
            ]))
        return OpenAI(
                client=OpenAIClient(api_key=api_key),
                model="gpt-3.5-turbo",
                exponential_backoff=True, chat=True,
                prompt=cfg["CLUSTER_REPRESENTATION_GPT_PROMPT"],
                generator_kwargs={
                    "seed": cfg["RANDOM_STATE"],
                    "temperature": 0,
                },
            )
        
    # BERTopic default, which is a sequence of top-n keywords
    else:
        return None
    
     
def plot_model_comparison(metrics: dict, output_path: str, fig_title: str):
    """ Generate a comparison plot between models, based on how model embeddings
        produce good clusters
    """
    # Function to keep only what will be plotted
    def norm_fn(d: dict[str, dict], dept_key: str) -> dict:
        to_plot = [
            # "Silhouette score", "DB index", "Dunn index", "MI score",
            "AMI score",  # "Homogeneity", "Completeness", "V measure",
        ]  # only AMI score because of many possible true label classes
        d_free, d_dept = d["label_free"], d["label_%s" % dept_key]
        d_free = {k: v for k, v in d_free.items() if k in to_plot}
        d_dept = {k: d_dept[k] for k in d_dept.keys() if k in to_plot}
        return d_dept
    
    # Select data to plot
    to_plot = {}
    for (model_name, metric) in metrics.items():
        to_plot[model_name] = norm_fn(metric, "dept")
    to_plot["rand"] = norm_fn(list(metrics.values())[0], "rand")
    to_plot["ceil"] = norm_fn(list(metrics.values())[0], "ceil")
    
    # Retrieve metric labels and model names
    labels = list(next(iter(to_plot.values())).keys())
    num_models = len(to_plot.keys())
    width = 0.8 / num_models  # Adjust width based on number of models
    
    # Plot metrics for each model
    fig, ax = plt.subplots(figsize=(3 * len(to_plot), 5))
    for idx, (model_name, metric) in enumerate(to_plot.items()):
        
        # Plot metric values
        x_values = [i + idx * width for i, _ in enumerate(labels)]
        y_values = list(metric.values())
        rects = ax.bar(x_values, y_values, width, label=model_name)
        
        # Auto-label the bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                text="{:.4f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom",
                rotation=90,
            )
            
    # Adjustments to the plot
    ax.set_title(fig_title, fontsize="x-large")
    ax.set_ylim(0.0, 1.25 * max(to_plot["ceil"].values()) if "ceil" in to_plot.keys() else 0.2)
    ax.set_xticks([i + width * (num_models - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(labels, fontsize="large", rotation=22.5)
    ax.set_ylabel("Scores", fontsize="x-large")
    ax.legend(fontsize="large", loc="upper right", ncol=1)
    ax.plot([-0.1, len(metrics) - 0.1], [0, 0], color="k")
    
    # Save final figure and csv file
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    df = pd.DataFrame.from_dict(to_plot, orient="index")
    df.to_csv(output_path.replace(".png", ".csv"))

    
if __name__ == "__main__":
    run_experiment_1()
    