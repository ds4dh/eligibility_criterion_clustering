import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, pearsonr
from statsmodels.stats.multitest import multipletests


DATA_DIR = "experiments/experiment_3_results"
FILTER_LEVELS = [{"cond": 4, "itrv": 3}]
EVALUATED_CT_COND_TYPE_FILTERS = ["C01", "C04", "C14", "C20"]
DATA_FILE_NAME = lambda t, c, i: "model-pubmed-bert-sentence_type%s_cond-%1i_itrv-%1i.csv" % (t, c, i)
METRICS = ["ROUGE-Average-F Score", "SciBERT-F Score"]
SUPPL_METRICS_C = [
    "ROUGE-1-F Score", "ROUGE-2-F Score", "ROUGE-L-F Score",
    "BERT-F Score", "SciBERT-F Score", "Longformer-F Score",
]
SUPPL_METRICS_D = [
    "ROUGE-Average-R Score", "ROUGE-1-Recall Score", "ROUGE-2-Recall Score", "ROUGE-L-Recall Score",
    "BERT-Recall Score", "SciBERT-Recall Score", "Longformer-Recall Score",
]
SUPPL_METRICS_E = [
    "ROUGE-Average-P Score", "ROUGE-1-Precision Score", "ROUGE-2-Precision Score", "ROUGE-L-Precision Score",
    "BERT-Precision Score", "SciBERT-Precision Score", "Longformer-Precision Score",
]
SUPPL_METRICS_CORRELATION_PLOT = [
    ["ROUGE-Average-F Score", "Quality"],
    ["Shuffled ROUGE-Average-F Score", "Quality"],
    ["SciBERT-F Score", "Quality"],
    ["Shuffled SciBERT-F Score", "Quality"],
]
MIN_Y_VALUES = {
    "SciBERT-F Score": 0.35,
}
MAX_Y_VALUES = {
    "ROUGE-Average-F Score": 0.35,
    "ROUGE-1-F Score": 0.55,
    "ROUGE-2-F Score": 0.20,
    "ROUGE-L-F Score": 0.25,
    "BERT-F Score": 0.75,
    "SciBERT-F Score": 0.75,
    "Longformer-F Score": 0.90,
}
METHOD_COLORS = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
METHODS = ["Cluster", "Shuffled Cluster", "LLM", "Shuffled LLM"]
LABEL_MAP = {
    "Quality": "Cluster Silhouette Score",
    "ROUGE-Average-F Score": "ROUGE-Average F1-Score",
    "ROUGE-Average-R Score": "ROUGE-Average Recall",
    "ROUGE-Average-P Score": "ROUGE-Average Precision",
    "Shuffled ROUGE-Average-F Score": "ROUGE-Average F1-Score - random baseline",
    "SciBERT-F Score": "F-BERT (SciBERT)",
    "SciBERT-Recall Score": "Recall-BERT (SciBERT)",
    "SciBERT-Precision Score": "Precision-BERT (SciBERT)",
    "Shuffled SciBERT-F Score": "F-BERT (SciBERT) - random baseline",
    "ROUGE-1-F Score": "ROUGE-1 F1-Score",
    "ROUGE-1-Recall Score": "ROUGE-1 Recall",
    "ROUGE-1-Precision Score": "ROUGE-1 Precision",
    "ROUGE-2-F Score": "ROUGE-2 F1-Score",
    "ROUGE-2-Recall Score": "ROUGE-2 Recall",
    "ROUGE-2-Precision Score": "ROUGE-2 Precision",
    "ROUGE-L-F Score": "ROUGE-L F1-Score",
    "ROUGE-L-Recall Score": "ROUGE-L Recall",
    "ROUGE-L-Precision Score": "ROUGE-L Precision",
    "BERT-F Score": "F-BERT (BERT)",
    "BERT-Recall Score": "Recall-BERT (BERT)",
    "BERT-Precision Score": "Precision-BERT (BERT)",
    "SciBERT-F Score": "F-BERT (SciBERT)",
    "SciBERT-Recall Score": "Recall-BERT (SciBERT)",
    "SciBERT-Precision Score": "Precision-BERT (SciBERT)",
    "Longformer-F Score": "F-BERT (Longformer)",
    "Longformer-Recall Score": "Recall-BERT (Longformer)",
    "Longformer-Precision Score": "Precision-BERT (Longformer)",
}
COND_TYPE_MAP = {
    "C01": "C01 - Infections",
    "C04": "C04 - Neoplasms",
    "C14": "C14 - Cardiovascular Diseases",
    "C20": "C20 - Immune System Diseases",
}


def main():
    """ Plot all figures, main one, and supplementary ones, about the generative
        task to evaluate information available in EC cluster information
    """
    for filter_level in FILTER_LEVELS:
        plot_one_metric_set(
            metric_set=METRICS,
            cond_types=EVALUATED_CT_COND_TYPE_FILTERS,
            filter_level=filter_level,
            output_path="experiments/figure_experiment_3.png",
            pool_cond_types=True,
        )
        plot_one_metric_set(
            metric_set=METRICS,
            cond_types=EVALUATED_CT_COND_TYPE_FILTERS,
            filter_level=filter_level,
            output_path="experiments/figure_experiment_3_suppl_B.png",
        )
        plot_one_metric_set(
            metric_set=SUPPL_METRICS_C,
            cond_types=EVALUATED_CT_COND_TYPE_FILTERS,
            filter_level=filter_level,
            output_path="experiments/figure_experiment_3_suppl_C.png",
        )
        plot_one_metric_set(
            metric_set=SUPPL_METRICS_D,
            cond_types=EVALUATED_CT_COND_TYPE_FILTERS,
            filter_level=filter_level,
            output_path="experiments/figure_experiment_3_suppl_D.png",
        )
        plot_one_metric_set(
            metric_set=SUPPL_METRICS_E,
            cond_types=EVALUATED_CT_COND_TYPE_FILTERS,
            filter_level=filter_level,
            output_path="experiments/figure_experiment_3_suppl_E.png",
        )
        plot_correlation(
            metric_pairs_set=SUPPL_METRICS_CORRELATION_PLOT,
            filter_level=filter_level,
            pooled_only=True,
            output_path="experiments/figure_experiment_3_suppl_F.png",
        )


def plot_one_metric_set(
    metric_set: list[str],
    cond_types: list[str],
    filter_level: dict[str, int],
    output_path: str,
    pool_cond_types: bool=False,
):
    """ Generate a comparison plot between cluster and LLM EC generation methods,
        for a given set of metrics
    """
    # Plot results pooling condition types
    if pool_cond_types:
        _, axs = plt.subplots(
            nrows=1, ncols=len(metric_set),
            figsize=(6 * len(metric_set), 6),
        )
        for i, metric in enumerate(metric_set):
            plot_one_metric(
                metric=metric,
                ax=axs[i],
                cond_type=cond_types,
                filter_level=filter_level,
                add_y_axis_label=False,
                add_condition_title=True,
                verbose=("suppl" not in output_path),
                num_comparisons=8,
            )
            
    # Plot results stratifying condition types
    else:
        _, axs = plt.subplots(
            nrows=len(metric_set), ncols=len(cond_types),
            figsize=(6 * len(cond_types), min(6 * len(metric_set), 30)),
        )
        if len(metric_set) == 1: axs = [axs]
        for i, metric in enumerate(metric_set):
            for j, cond_type in enumerate(cond_types):
                plot_one_metric(
                    metric=metric,
                    ax=axs[i][j],
                    cond_type=cond_type,
                    filter_level=filter_level,
                    num_comparisons=1,
                    add_y_axis_label=(j == 0),
                    add_condition_title=(i == 0),
                    verbose=("suppl" not in output_path),
                )
                
    # Save adjusted plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    
def plot_one_metric(
    metric: str,
    ax: "plt.Axis",
    cond_type: str|list[str],
    filter_level: dict[str, int],
    add_y_axis_label: bool,
    add_condition_title: bool,
    verbose: bool,
    num_comparisons: int=None,
) -> float:
    """ Plot results of experiment 3 for one set of condition and intervention levels
    """
    # Load the data
    plotted_keys = ["%s %s" % (method, metric) for method in METHODS]
    ceiling_key = "Ceiling Cluster %s" % metric  # could be "Ceiling LLM %s", it is the same
    if isinstance(cond_type, list):
        csv_file_paths = [DATA_FILE_NAME(c, filter_level["cond"], filter_level["itrv"]) for c in cond_type]
        df_raw_list = [pd.read_csv(os.path.join(DATA_DIR, p)) for p in csv_file_paths]
        df_raw = pd.concat(df_raw_list, ignore_index=True)
    else:
        csv_file_path = DATA_FILE_NAME(cond_type, filter_level["cond"], filter_level["itrv"])
        df_raw = pd.read_csv(os.path.join(DATA_DIR, csv_file_path))
        
    # Filter samples where the clustering algorithm converged
    df = df_raw[df_raw["Cluster Quality"] != -1.0]  # if clustering did not converge (but there are none)
    mean_converging = df[plotted_keys].mean()
    stderr_converging = df[plotted_keys].std() / np.sqrt(len(df))
    mean_ceiling = df[ceiling_key].mean()
    stderr_ceiling = df[ceiling_key].std() / np.sqrt(len(df))  # probably will not use
    
    # Print some useful values
    if verbose: 
        plotted_metric = plotted_keys[0].split("Cluster ")[-1]
        cluster_over_llm = mean_converging["Cluster " + plotted_metric] / mean_converging["LLM " + plotted_metric] * 100
        cluster_diff_random = mean_converging["Cluster " + plotted_metric] - mean_converging["Shuffled Cluster " + plotted_metric]
        llm_diff_random = mean_converging["LLM " + plotted_metric] - mean_converging["Shuffled LLM " + plotted_metric]
        cluster_diff_over_llm_diff = cluster_diff_random / llm_diff_random * 100
        print("Evaluating %s for condition(s) %s" % (metric, cond_type))
        print("Cluster makes up %4f%% of LLM performance" % cluster_over_llm)
        print("Cluster vs random makes up %4f%% of LLM vs random" % cluster_diff_over_llm_diff)
    
    # Initialize bar plot utilities
    max_value = 0.0
    bar_width = 0.16  # should be greater than 0.0 and smaller than 0.25
    def pos_fn(idx):
        group_shift = (idx // 2 - 0.5) * 2 * (1 + 2 * bar_width) / 6
        bar_shift = (idx % 2 - 0.5) * bar_width
        return 0.5 + group_shift + bar_shift
    
    # Plot the bars
    ax_handles = []
    ax_labels = []
    for i, column in enumerate(plotted_keys):
        color = METHOD_COLORS[i % len(METHODS)]
        label = METHODS[i % len(METHODS)] if i < len(METHODS) else None
        handle = ax.bar(
            pos_fn(i), mean_converging[column], bar_width,
            yerr=stderr_converging[column], alpha=0.75, label=label, color=color,
            capsize=4, error_kw={"elinewidth": 2, "capthick":2},
        )
        if label not in ax_labels:
            ax_handles.append(handle)
            ax_labels.append(label)
        
        # Statistical comparisons
        if num_comparisons is not None:
            significant = lambda p: "*" if p < 0.05 else "n.s."
            significant_bf = lambda p: "*" if p < 0.05 / num_comparisons else "n.s."
            
            # Statistical comparison (normal vs shuffled, for both cluster and LLM)
            if "Shuffled" not in column:
                column_comp = "Shuffled %s" % column
                _, p_value_vs_rand = ttest_rel(
                    df[plotted_keys][column],
                    df[plotted_keys][column_comp],
                )
                significance = significant(p_value_vs_rand)
                significance_bf = significant_bf(p_value_vs_rand)
                if verbose:
                    print(
                        "P-value of %s vs %s (paired t-test, %i samples): %f (%s, BF-%s)"\
                        % (column, column_comp, len(df), p_value_vs_rand, significance, significance_bf)
                        )
            
            # Statistical comparison (cluster vs LLM, for both normal and random)
            if "Cluster" in column:
                column_comp = column.replace("Cluster", "LLM")
                _, p_value_vs_llm = ttest_rel(
                    df[plotted_keys][column],
                    df[plotted_keys][column_comp],
                )
                significance = significant(p_value_vs_llm)
                significance_bf = significant_bf(p_value_vs_llm)
                if verbose:
                    print(
                        "P-value of %s vs %s (paired t-test, %i samples): %f (%s, BF-%s)" %\
                        (column, column_comp, len(df), p_value_vs_llm, significance, significance_bf)
                    )
        
        # Add individual data points with jitter
        jitter = bar_width / 3 * (np.random.rand(len(df)) - 0.5)
        ax.scatter(
            np.full(len(df), pos_fn(i)) + jitter,
            df[column], color="k", alpha=0.1, s=2,
        )
        this_max = max(df[column].max(), df[column].max())
        if this_max > max_value: max_value = this_max
    if verbose: print("\n", end="")
    
    # Plot ceiling performance
    ax.plot(
        [pos_fn(0) - bar_width, pos_fn(i) + bar_width], 
        [mean_ceiling, mean_ceiling], linestyle="--", linewidth=2,
        label="Ceiling" if i == 0 else "", color="black",
    )
    
    # Polish the plot
    def tick_label_fn(m):
        return "\n".join(m.split(" ")[::-1]).replace("Shuffled", "Random")
    x_tick_labels = [tick_label_fn(m) for m in METHODS]
    ax.set_xticks([pos_fn(i) for i in range(4)], x_tick_labels, fontsize=14)
    ax.set_xlim([0, 1])
    bottom_x = MIN_Y_VALUES.get(metric, 0.0)
    top_x = 1.1 * mean_ceiling - 0.1 * MIN_Y_VALUES.get(metric, 0.0)
    ceil_text_pos_x = 0.5
    ceil_text_pos_y = mean_ceiling - (top_x - bottom_x) * 0.05
    if "longformer" in metric.lower():
        ceil_text_pos_y = mean_ceiling + (top_x - bottom_x) * 0.04
    ax.set_ylim(bottom=bottom_x, top=top_x)
    ax.text(
        ceil_text_pos_x, ceil_text_pos_y, "Ceiling performance",
        color="black", fontsize=16, ha="center", va="center",
    )
    if add_y_axis_label:
        ax.set_ylabel(LABEL_MAP[metric], fontsize=16)
    if add_condition_title:
        if isinstance(cond_type, list):
            ax.set_title(LABEL_MAP[metric], fontsize=16)
        else:
            ax.set_title(COND_TYPE_MAP[cond_type], fontsize=16)


def plot_correlation(
    metric_pairs_set: list[str],
    filter_level: dict[str, int],
    pooled_only: bool=False,
    output_path: str=None,
    axs: "list[plt.Axis]"=None,
) -> None:
    """ Plot correlation between cluster quality and EC section generation
        quality, using Silhouette Score and Rouge-Average F1-Score
    """
    # Check everything is smooth
    print("Evaluating correlation between cluster quality and eligibility section generation")
    assert not (output_path is None and axs is None),\
        "Either ax or output_path should be given"
    assert not (output_path is not None and axs is not None),\
        "Only one of ax or output_path can be given"
    
    # Load and filter data for each condition type
    def path_fn(t: str) -> str:
        return os.path.join(
            DATA_DIR,
            DATA_FILE_NAME(t, filter_level["cond"], filter_level["itrv"]),
        )
    df_dict = {
        cond_type: pd.read_csv(path_fn(cond_type))
        for cond_type in EVALUATED_CT_COND_TYPE_FILTERS
    }
    
    # Arrange data in a metric-pair dict of cond-type dicts
    metric_pair_dict = {}
    for metric_pair in metric_pairs_set:
        assert len(metric_pair) == 2,\
            "Metric set should contain 2 correlated metrics"
        
        plotted_keys = ["Cluster %s" % (metric) for metric in metric_pair]
        plotted_keys = [k.replace("Cluster Shuffled", "Shuffled Cluster") for k in plotted_keys]
        df_dict_plotted = {k: df[plotted_keys] for k, df in df_dict.items()}
        
        if pooled_only:
            df_dict_plotted = {"All": pd.concat(df_dict_plotted, ignore_index=True)}
        else:
            df_dict_plotted["All"] = pd.concat(df_dict_plotted, ignore_index=True)
        
        metric_pair_key = "***".join(plotted_keys)
        metric_pair_dict[metric_pair_key] = df_dict_plotted
        
    # Initialize plot if required
    if axs is None:
        n_rows = len(metric_pairs_set)
        n_cols = len(df_dict_plotted)  # last one added (same for all)
        if pooled_only: n_rows, n_cols = n_cols, n_rows
        _, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            figsize=(6 * n_cols, 6 * n_rows),
        )
    else:
        assert len(axs) == len(df_dict_plotted),\
            "Ensure there are as many axes provided as plotted metrics"
    
    # Initialize correlation and p-values matrix for new plot
    corr_matrix = np.zeros((len(metric_pairs_set), len(df_dict_plotted)))
    p_matrix = np.ones((len(metric_pairs_set), len(df_dict_plotted)))
    
    # Plot correlation for each combination of metric and condition type
    if len(metric_pairs_set) == 1 or pooled_only: axs = [axs]
    for i, (metric_pair_key, df_plotted) in enumerate(metric_pair_dict.items()):
        for j, (cond_type, df) in enumerate(df_plotted.items()):
            
            # Gather information
            x_key, y_key = metric_pair_key.split("***")
            x_label = "".join(x_key.split("Cluster "))  # [-1]
            y_label = "".join(y_key.split("Cluster "))  # [-1]
            
            # Compute correlation
            x = df[x_key].values
            y = df[y_key].values
            corr, p_value = pearsonr(x, y)
            
            # Store correlation and p-value in matrices
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_value
            
            # Select order in which plots are arranged
            if not pooled_only:
                selected_ax = axs[i][j]
            else:
                selected_ax = axs[j][i]
            
            # Scatter plot
            selected_ax.scatter(x, y, s=10)
            if not pooled_only:
                selected_ax.set_title(f"{cond_type} (r={corr:.4f})", fontsize=16)
            else:
                selected_ax.set_title(
                    f"Cluster vs generation quality (r={corr:.4f})",
                    fontsize=16,
                )
            selected_ax.set_xlabel(LABEL_MAP[x_label], fontsize=14)
            selected_ax.set_ylabel(LABEL_MAP[y_label], fontsize=14)
            
            # Print the correlation and significance result
            if cond_type == "All":
                significance = "significant" if p_value < 0.05 else "not significant"
                print(
                    "%s vs %s Pearson correlation (r=%.4f), p-value=%.4f -> %s" %\
                    (y_key, x_key, corr, p_value, significance)
                )
                
    # Adjust layout and save the figure if required
    if output_path is not None:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        # save_correlation_matrix(
        #     p_matrix=p_matrix,
        #     corr_matrix=corr_matrix,
        #     output_path=output_path.replace(".png", "_matrix.png"),
        # )


def save_correlation_matrix(
    p_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    output_path: str,
) -> None:
    """ Create a correlation matrix with ovals and bf-corrected significance stars
    """
    # Heatmap for correlation values
    _, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        data=corr_matrix, annot=False,
        cmap="RdBu", center=0, ax=ax,
        cbar_kws={"label": "Correlation coefficient"}
    )
    
    # Add stars for significance
    p_adjusted = multipletests(
        pvals=p_matrix.flatten(),
        method="bonferroni",
    )[1].reshape(p_matrix.shape)
    for i in range(p_adjusted.shape[0]):
        for j in range(p_adjusted.shape[1]):
            if p_adjusted[i, j] < 0.05:  # BF-corrected significance level
                ax.text(
                    j + 0.5, i + 0.5, "*", color="black",
                    fontsize=20, ha="center", va="center"
                )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    main()
    