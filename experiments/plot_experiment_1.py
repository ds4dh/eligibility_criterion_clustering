import matplotlib.axes
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


DATA_DIR = "data_ctgov"
OUTPUT_PATH_MAIN = "experiments/figure_experiment_1.png"
OUTPUT_PATH_SUPPL = "experiments/figure_experiment_1_suppl.png"
SUB_DIR_A1 = "cond-lvl-4_itrv-lvl-3_cluster-tsne-2_plot-tsne-2"
SUB_DIR_A2 = "cond-lvl-3_itrv-lvl-2_cluster-tsne-2_plot-tsne-2"
SUB_DIR_A3 = "cond-lvl-2_itrv-lvl-1_cluster-tsne-2_plot-tsne-2"
TITLE_A1 = "Condition label MeSH level: 4, Intervention label MeSH level: 3"
TITLE_A2 = "Condition label MeSH level: 3, Intervention label MeSH level: 2"
TITLE_A3 = "Condition label MeSH level: 2, Intervention label MeSH level: 1"
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
    "ceil": "Ceiling",
}
MODEL_ORDER = [
    "PubMed-BERT-Sentence",
    "BERT-Sentence",
    "PubMed-BERT",
    "BERT",
    "Random",
]


def main():
    """ Plot results of experiment 1 in a 2-panel figure, and an alternative
        version of figure 1A in 
    """
    # Main-text Figure 1
    plot_figure_1(
        output_path=OUTPUT_PATH_MAIN,
        add_figure_1b=True,
        normalize_ceil_perf=True,
    )
    
    # Supplementary versino of Figure 1
    plot_figure_1(
        output_path=OUTPUT_PATH_SUPPL,
        add_figure_1b=False,
        normalize_ceil_perf=False,
    )


def plot_figure_1(
    output_path: str,
    add_figure_1b: bool=True,
    normalize_ceil_perf: bool=True,
):
    """ Plot results for Figure 1, with or without Figure 1B, with or without
        normalizing ceiling performance in Figure 1A
    """
    # Create a gridspec figure to have 3 rows on the left, 1 column on the right
    figure_width = 0.5 * len(MODEL_ORDER) * len(COND_IDS) + 3 * add_figure_1b
    width_ratios = [6.5, 1] if add_figure_1b else None
    num_cols = 2 if add_figure_1b else 1
    fig = plt.figure(figsize=(figure_width, 12))
    gs = GridSpec(3, num_cols, figure=fig, width_ratios=width_ratios)
    ax_A1 = fig.add_subplot(gs[0, 0])  # first row, first column
    ax_A2 = fig.add_subplot(gs[1, 0])  # second row, first column
    ax_A3 = fig.add_subplot(gs[2, 0])  # third row, first column
    if add_figure_1b:
        ax_B = fig.add_subplot(gs[:, 1])  # all rows, second column
    
    # Plot experiments in the relevant gridspec areas
    plot_experiment_1A(SUB_DIR_A1, ax_A1, normalize_ceil_perf, title=TITLE_A1)
    plot_experiment_1A(SUB_DIR_A2, ax_A2, normalize_ceil_perf, title=TITLE_A2)
    plot_experiment_1A(SUB_DIR_A3, ax_A3, normalize_ceil_perf, title=TITLE_A3)
    if add_figure_1b:
        plot_experiment_1B(ax_B)
    
    # Save polished supplementary figure
    plt.tight_layout(h_pad=2.0, w_pad=3.0)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    
def plot_experiment_1A(
    sub_dir: str,
    ax: matplotlib.axes.Axes,
    normalize_by_ceiling: bool,
    title: str|None=None,
):
    """ Plot results of experiment 1A with grouped bars for each condition
    """
    # Generate a list of csv file paths
    csv_file_paths = [
        f"{DATA_DIR}/ctgov-{cond_id}/{sub_dir}/results/model-comparison.csv"
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
        ceil_perf = df[df["Model"] == "Ceiling"]["AMI score"].item()
        df = df.set_index("Model").reindex(MODEL_ORDER).reset_index()
        scores = df["AMI score"] + 0.001  # to see the random column (close to 0)
        
        # Plot each model's bar
        for j, score in enumerate(scores):
            if normalize_by_ceiling: score = score / ceil_perf
            ax.bar(
                index[i] + j * bar_width, score, bar_width, alpha=0.75,
                label=MODEL_ORDER[j] if i == 0 else "", color=MODEL_COLORS[j],
            )
        
        # Plot different ceiling performance for each subgraph
        if not normalize_by_ceiling:
            ax.plot(
                [index[i] - 0.5 * bar_width, index[i] + (len(MODEL_ORDER) - 0.5) * bar_width], 
                [ceil_perf, ceil_perf], linestyle="--", linewidth=2.0,
                label="Ceiling" if i == 0 else "", color="black",
            )
        else:
            ceil_perf = 1.0
                
        # Print / record some values
        if ceil_perf * 1.1 > max_value: max_value = ceil_perf * 1.1
        print("\nCondition %s (ceiling perf: %.3f)" % (COND_IDS[i], ceil_perf))
        print(df, end="\n\n" if i == len(COND_IDS) - 1 else "\n")
    
    # Plot common ceiling performance if normalized
    if normalize_by_ceiling:
        ax.plot(
            [-bar_width, len(COND_IDS) - 1 + (len(MODEL_ORDER)) * bar_width], 
            [ceil_perf, ceil_perf], linestyle="--", linewidth=2.0,
            label="Ceiling", color="black",
        )
        
    # Set labels and title
    ax.set_ylabel("Normalized AMI Score" if normalize_by_ceiling else "AMI Score", fontsize=16)
    ax.set_ylim([0.0, max_value * 1.25])
    ax.set_xticks(index + bar_width * (len(scores) - 1) / 2)
    ax.set_xticklabels(COND_IDS, fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    if title is not None: ax.set_title(title, fontsize=16)
    
    # Plot legend in the wanted order
    handles, labels = ax.get_legend_handles_labels()
    ceiling_index = labels.index("Ceiling")
    handles = handles[:ceiling_index] + handles[ceiling_index+1:] + [handles[ceiling_index]]
    labels = labels[:ceiling_index] + labels[ceiling_index+1:] + [labels[ceiling_index]]
    ax.legend(handles, labels, ncol=len(MODEL_ORDER) // 2 + 1, fontsize=14, loc="upper center")


def plot_experiment_1B(ax):
    """ Plot results of experiment 1B with vertically stacked bars for each model
    """
    # Load dataframe from excel file
    excel_file = pd.ExcelFile("./experiments/experiment_1B_results/raw_data.xlsx")
    df = excel_file.parse(excel_file.sheet_names[0])
    
    # Extract columns and rows of interest
    plot_index = df.index[df["case 1, sort"] == "case 3, sort"][0]
    plot_data = df.loc[
        plot_index + 1:plot_index + 3,
        ["Unnamed: 0", "case 1, sort"],
    ].set_index("Unnamed: 0")
    plot_data.columns = ["PubMed-BERT-\nSentence"]
    plot_data = plot_data / plot_data.sum() * 100
    
    # Plot as a stacked bar plot
    plot_data = plot_data.T  # transpose data for stacking
    plot_data = plot_data[["yes", "other", "no"]]  # to have order correct, unclear, incorrect
    plot_data.plot(
        kind="bar", stacked=True, alpha=0.75, ax=ax,
        color=["green", "gray", "firebrick"],
    )
    
    # Polish plot
    ax.set_ylim([0, 115])
    ax.set_title("Proportion (%)", fontsize=16)
    ax.set_xticks(range(len(plot_data.index)))
    ax.set_xticklabels(plot_data.index, rotation=0, fontsize=16)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(
        ["Correct", "Unclear", "Incorrect"],
        fontsize=14, loc="upper center", handletextpad=0.25,
    )


def plot_experiment_1B_old(ax):
    """ Plot results of experiment 1B with grouped bars for each model
    """
    # Load dataframe from excel file
    excel_file = pd.ExcelFile("./experiments/experiment_1B_results/raw_data.xlsx")
    df = excel_file.parse(excel_file.sheet_names[0])
    
    # Extract columns and rows of interest
    case3_index = df.index[df["case 1, sort"] == "case 3, sort"][0]
    case3_data = df.loc[
        case3_index + 1:case3_index + 3,
        ["Unnamed: 0", "case 1, sort"],
    ].set_index("Unnamed: 0")
    case3_data.columns = ["PubMed-BERT-Sentence"]
    case3_data = case3_data / case3_data.sum() * 100
    
    # Plot results
    df_plot = case3_data.T
    df_plot.plot(kind="bar", alpha=0.75, ax=ax)
    
    # Polish plot
    ax.set_ylim([0, 100])
    ax.set_ylabel("Proportion (%)", fontsize=16, labelpad=0)
    ax.set_xticks(range(len(df_plot.index)))
    ax.set_xticklabels(df_plot.index, rotation=0, fontsize=16)
    ax.legend(["Correct", "Incorrect", "Unclear"], fontsize=14)

    
if __name__ == "__main__":
    main()
    