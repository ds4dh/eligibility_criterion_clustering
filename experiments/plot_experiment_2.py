import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import ttest_rel


OUTPUT_PATH = "experiments/figure_experiment_2.png"
DATA_DIR = "data_ctgov"
MODEL_ID = "pubmed-bert-sentence"
SUB_DIR = "cond-lvl-4_itrv-lvl-3_cluster-tsne-2_plot-tsne-2"  # same as for experiment 1
COND_IDS = ["C01", "C04", "C14", "C20"]
INPUT_COLORS = ["tab:blue", "tab:orange", "tab:green"]
TARGET_TYPES = ["phase", "study_duration", "enrollment_count", "operational_rate"]
INPUT_TYPES = ["random", "cluster_ids", "raw_embeddings"]


def main():
    """ Plot results of experiment 2 with grouped bars for each input type
    """
    pooled_results = defaultdict(dict)
    for target_type in TARGET_TYPES:
        for input_type in INPUT_TYPES:
            pooled_csv_file_paths = []
            for cond_id in COND_IDS:
                result_dir = "%s/ctgov-%s/%s/results/%s/predict_results" % (
                    DATA_DIR, cond_id, SUB_DIR, MODEL_ID,
                )
                csv_file_paths = [
                    os.path.join(result_dir, p) for p in os.listdir(result_dir)
                    if p.endswith(".csv")\
                    and ("T" + target_type) in p\
                    and ("I" + input_type) in p
                ]
                pooled_csv_file_paths.extend(csv_file_paths)
                
            # Result pooling all condition ids for a particular set of target and input type
            pooled_results[target_type][input_type] = extract_macro_avg_f1(pooled_csv_file_paths)
            
    # Plot result comparison
    plot_pooled_comparison(pooled_results)
                

def extract_macro_avg_f1(file_paths):
    """ Extract the mean and standard error for smacro average f1-scores from a
        list of CSV files
    """
    f1_scores = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        macro_avg_f1 = df[df["Unnamed: 0"] == "macro avg"]["f1-score"].values[0]
        f1_scores.append(macro_avg_f1)
    
    mean_f1 = np.mean(f1_scores)
    if len(f1_scores) > 1:
        stderr_f1 = np.std(f1_scores, ddof=1) / np.sqrt(len(f1_scores))
    else:
        stderr_f1 = 0.0
        
    return {"mean": mean_f1, "stderr": stderr_f1, "values": f1_scores}


def plot_pooled_comparison(results):
    """ Plots a comparison of mean values with error bars and individual values
    """
    # Initialize plot
    bar_width = 0.2
    max_value = 0.0
    index = np.arange(len(TARGET_TYPES))
    _, ax = plt.subplots(figsize=(2.5 * len(TARGET_TYPES), 5.5))
    
    # Statistical tests utilities
    num_comparisons = (len(TARGET_TYPES) * len(INPUT_TYPES))
    significant = lambda p: "*" if p < 0.05 else "n.s."
    significant_bf = lambda p: "*" if p < 0.05 / num_comparisons else "n.s."
    
    # Iterate over input types and plot the bars
    for i, input_type in enumerate(INPUT_TYPES):
        means = [results[target_type][input_type]["mean"] for target_type in TARGET_TYPES]
        stderrs = [results[target_type][input_type]["stderr"] for target_type in TARGET_TYPES]
        
        # Plot the bars with error bars
        label = input_type.capitalize().replace("_", " ")
        ax.bar(
            index + i * bar_width, means, bar_width, label=label, yerr=stderrs,
            alpha=0.75, capsize=4, error_kw={"elinewidth": 2, "capthick":2},
        )
        
        # Plot using scatter with jitter
        for j, target_type in enumerate(TARGET_TYPES):
            values = results[target_type][input_type]["values"]
            jitter = np.random.uniform(-bar_width/4, bar_width/4, size=len(values))
            ax.scatter(
                np.full(len(values), index[j] + i * bar_width) + jitter,
                values, color="k", alpha=0.3, s=4,
            )
            if max(values) > max_value: max_value = max(values)
    
            # Print statistical test results
            if input_type != "random":
                _, p_value_vs_rand = ttest_rel(
                    results[target_type][input_type]["values"],
                    results[target_type]["random"]["values"],
                )
                significance = significant(p_value_vs_rand)
                significance_bf = significant_bf(p_value_vs_rand)
                num_samples = len(results[target_type][input_type]["values"])
                print(
                    "P-value of %s vs random for %s (paired t-test, %i samples): %f (%s, BF-%s)"\
                    % (input_type, target_type, num_samples, p_value_vs_rand, significance, significance_bf)
                )
                if input_type == "cluster_ids":
                    _, p_value_vs_raw = ttest_rel(
                        results[target_type][input_type]["values"],
                        results[target_type]["raw_embeddings"]["values"],
                    )
                    significance = significant(p_value_vs_raw)
                    significance_bf = significant_bf(p_value_vs_raw)
                    num_samples = len(results[target_type][input_type]["values"])
                    print(
                        "P-value of %s vs raw_embeddings for %s (paired t-test, %i samples): %f (%s, BF-%s)"\
                        % (input_type, target_type, num_samples, p_value_vs_raw, significance, significance_bf)
                    )
                    
        # Print useful differences
        if input_type != "random":
            diff_with_rand = [
                means[j] - results[target_type]["random"]["mean"]
                for j, target_type in enumerate(TARGET_TYPES)
            ]
            print("%s mean value: %f" % (input_type, np.mean(means)))
            print("%s diff with random: %s" % (input_type, np.mean(diff_with_rand)))
            
        # Print exact values
        print("\nInput_type:")
        print(input_type)
        print("Means:")
        print(means)
        print("Stderr:")
        print(stderrs, end="\n\n" if i == len(INPUT_TYPES) - 1 else "\n")
                
    # Polish plot
    ax.set_ylim([0, max_value * 1.15])
    ax.set_ylabel("Macro-averaged F1-score", fontsize=16)
    ax.set_xticks(index + bar_width)
    x_tick_labels = [t.capitalize().replace("_", " ") for t in TARGET_TYPES]
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.legend(fontsize=14, ncol=len(INPUT_TYPES), loc="upper center")
    
    # Adjust layout and save plot
    os.makedirs(os.path.split(OUTPUT_PATH)[0], exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    
    
if __name__ == "__main__":
    main()
    