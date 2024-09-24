import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_PATH = "experiments/figure_experiment_1.png"
DATA_DIR = "data_ctgov"
SUB_DIR = "cond-lvl-4_itrv-lvl-3_cluster-tsne-2_plot-tsne-2"
COND_IDS = ["C01", "C04", "C14", "C20"]
COND_NAMES = {
    "C01": "Infections",
    "C04": "Neoplasms",
    "C14": "Cardiovascular Diseases",
    "C20": "Immune System Diseases",
}
MODEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MODEL_MAP = {
    "pubmed-bert-sentence": "PubMed-BERT-Sentence",
    "bert-sentence": "BERT-Sentence",
    "pubmed-bert-token": "PubMed-BERT",
    "bert": "BERT",
    "rand": "Random",
}
MODEL_ORDER = [
    "PubMed-BERT-Sentence",
    "BERT-Sentence",
    "PubMed-BERT",
    "BERT",
    "Random",
]


def main():
    """ Plot results of experiment 1A and 1B in a common 2-panel figure
    """
    # Create subplot
    _, axs = plt.subplots(
        nrows=1,
        ncols=2,
        width_ratios=(3, 1),
        figsize=(0.5 * len(MODEL_ORDER) * len(COND_IDS) + 6, 4.5),
    )
    
    # Plot each experiment on one ax
    plot_experiment_1A(axs[0])
    plot_experiment_1B(axs[1])
    
    # Save polished figure
    plt.subplots_adjust(wspace=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    
    
def plot_experiment_1A(ax):
    """ Plot results of experiment 1A with grouped bars for each condition
    """
    # Generate a list of csv file paths
    csv_file_paths = [
        f"{DATA_DIR}/ctgov-{cond_id}/{SUB_DIR}/results/model-comparison.csv"
        for cond_id in COND_IDS
    ]

    # Initialize the plot
    bar_width = 0.14
    max_value = 0.0
    index = np.arange(len(COND_IDS))
    
    # Load results from the csv file of each condition id
    for i, csv_file_path in enumerate(csv_file_paths):
        df = pd.read_csv(csv_file_path)
        df["Model"] = df.iloc[:, 0].map(MODEL_MAP)
        df = df.set_index("Model").reindex(MODEL_ORDER).reset_index()
        scores = df["AMI score"] + 0.001  # to see the random column (close to 0)
        if max(scores) > max_value: max_value = max(scores)
        
        # Plot each model"s bar
        for j, score in enumerate(scores):
            ax.bar(
                index[i] + j * bar_width, score, bar_width, alpha=0.75,
                label=MODEL_ORDER[j] if i == 0 else "", color=MODEL_COLORS[j],
            )
        
        # Print exact values
        print("\nCondition:")
        print(COND_IDS[i])
        print(df, end="\n\n" if i == len(COND_IDS) - 1 else "\n")

    # Set labels and title
    ax.set_ylim([0.0, max_value * 1.3])
    ax.set_ylabel("AMI Score", fontsize=16)
    ax.set_xticks(index + bar_width * (len(scores) - 1) / 2)
    ax.set_xticklabels(COND_IDS, fontsize=16)
    ax.legend(ncol=len(MODEL_ORDER) // 2 + 1, fontsize=14, loc="upper center")


def plot_experiment_1B(ax):
    """ Plot results of experiment 1B with grouped bars for each model
    """
    # Load dataframe from excel file
    excel_file = pd.ExcelFile("./experiments/experiment_1B_results/raw_data.xlsx")
    df = excel_file.parse(excel_file.sheet_names[0])
    
    # Extract columns and rows of interest
    case3_index = df.index[df["case 1, sort"] == "case 3, sort"][0]
    case3_data = df.loc[case3_index + 1:case3_index + 3, ["Unnamed: 0", "case 1, sort"]].set_index("Unnamed: 0")
    case3_data.columns = ["PubMed-BERT-Sentence"]
    case3_data = case3_data / case3_data.sum() * 100
    
    # Plot results
    df_plot = case3_data.T
    df_plot.plot(kind="bar", alpha=0.75, ax=ax)
    
    # Polish plot
    ax.set_ylim([0, 100])
    ax.set_ylabel("Accuracy (%)", fontsize=16, labelpad=0)
    ax.set_xticks(range(len(df_plot.index)))
    ax.set_xticklabels(df_plot.index, rotation=0, fontsize=16)
    ax.legend(["Correct", "Incorrect", "Unclear"], fontsize=14)
    
    
if __name__ == "__main__":
    main()
    