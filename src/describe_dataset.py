import os
import pandas as pd
from tqdm import tqdm
from typing import Any


DATA_DIR = "data_ctgov"
DATA_PATH = os.path.join(DATA_DIR, "parsed_criteria.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "descriptive_statistics.csv")
USED_DF_COLUMNS = ["individual criterion", "ct_path", "condition_ids", "intervention_ids"]
COND_ID_FILTERS = ["C%02i" % i for i in range(1, 30)]
ITRV_ID_FILTERS = ["D%02i" % i for i in range(1, 30)]


def main():
    # Load and format dataset
    ec_df = pd.read_csv(DATA_PATH, encoding="utf-8")
    for column in tqdm(ec_df.columns, "Formatting dataset"):
        ec_df[column] = ec_df[column].apply(eval_cleaning_fn)
    
    # Reset result file and record whole dataset statistics
    if os.path.exists(OUTPUT_PATH): os.remove(OUTPUT_PATH)
    print("Computing general statistics")
    n_ecs, n_cts = record_descriptive_statistics(ec_df, "Total")
    
    # Record statistics stratifying by condition ID type
    for cond_filter in tqdm(COND_ID_FILTERS, desc="Stratifying by condition ids"):
        filtered_ec_df = filter_by_ids(ec_df, cond_filter, "condition_ids")
        if len(filtered_ec_df) > 0:
            record_descriptive_statistics(filtered_ec_df, cond_filter, n_ecs, n_cts)
            
    # Record statistics stratifying by intervention ID type
    for itrv_filter in tqdm(ITRV_ID_FILTERS, desc="Stratifying by intervention ids"):
        filtered_ec_df = filter_by_ids(ec_df, itrv_filter, "intervention_ids")
        if len(filtered_ec_df) > 0:
            record_descriptive_statistics(filtered_ec_df, itrv_filter, n_ecs, n_cts)


def eval_cleaning_fn(expression: str) -> Any:
    """ Evaluate the dataframe values because they are all strings

    Args:
        expression (str): expression to evaluate (or not)

    Returns:
        Any: evaluated expression or expression itself
    """
    try:
        return eval(expression)
    except (NameError, SyntaxError, TypeError):
        return expression


def record_descriptive_statistics(
    ec_df: pd.DataFrame,
    split: str,
    total_ec_count: int=None,
    total_ct_count: int=None,
) -> None:
    """ Prints the main statistics of a dataframe

    Args:
        ec_df (pd.DataFrame): dataframe from which main statistics are shown
        split (str): type of split for the data being processed
    """
    # Count ECs and CTs
    ct_df = ec_df.drop_duplicates(subset="ct path", keep="first")
    n_ecs = len(ec_df)
    n_cts = len(ct_df)
    
    # Count ECs per CT
    criterion_counts = ec_df.groupby("ct path")["individual criterion"].count()
    avg_ec_per_ct = criterion_counts.mean()
    std_ec_per_ct = criterion_counts.std()
    
    # Count phases
    df_exploded = ec_df.explode("phases")
    ec_counts = df_exploded.groupby("phases")["individual criterion"].count().to_dict()
    ct_counts = df_exploded.groupby("phases")["ct path"].nunique().to_dict()
    rows = []
    if total_ec_count is None and total_ct_count is None:
        rows.append([split, "Total EC Count", n_ecs, "Total CT Count", n_cts])
    else:
        ec_prop = n_ecs / total_ec_count * 100
        ct_prop = n_cts / total_ct_count * 100
        ec_str = "%i (%.2f%%)" % (n_ecs, ec_prop)
        ct_str = "%i (%.2f%%)" % (n_cts, ct_prop)
        rows.append([split, "Total EC Count", ec_str, "Total CT Count", ct_str])
    for phase_id in [1, 2, 3, 4]:
        phase_ec_count = ec_counts.get("phase%1i" % phase_id, 0)
        phase_ct_count = ct_counts.get("phase%1i" % phase_id, 0)
        phase_ec_prop = phase_ec_count / n_ecs * 100
        phase_ct_prop = phase_ct_count / n_cts * 100
        phase_ec_title = "[TAB]Phase %1i EC Count" % phase_id
        phase_ct_title = "[TAB]Phase %1i CT Count" % phase_id
        phase_ec_str = "[TAB]%i (%.2f%%)" % (phase_ec_count, phase_ec_prop)
        phase_ct_str = "[TAB]%i (%.2f%%)" % (phase_ct_count, phase_ct_prop)
        rows.append(["", phase_ec_title, phase_ec_str, phase_ct_title, phase_ct_str])
    rows.append(["", "[TAB]ECs per CT", "[TAB]%.2f ± %.2f" % (avg_ec_per_ct, std_ec_per_ct), "", ""])
    
    # Append to the CSV, creating the file with headers if it doesn't exist
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(
        path_or_buf=OUTPUT_PATH,
        mode='a',
        header=not pd.io.common.file_exists(OUTPUT_PATH),
        index=False,
        encoding="utf-8",
    )
    
    # Return total count if used on the whole dataset
    if total_ec_count is None and total_ct_count is None:
        return n_ecs, n_cts
    
    
def filter_by_ids(
    df: pd.DataFrame,
    filter_str: str,
    filtered_column: str,
) -> pd.DataFrame:
    """ Filter an eligibility criteria dataset by whether a row includes or not
        a given ID in condition or intervention ID lists of lists
        
    Args:
        df (pd.DataFrame): dataframe to filter
        filter_str (str): filter to match
        filtered_column (str): column to look for a match
        
    Returns:
        pd.DataFrame: filtered dataframe
    """
    # Check if any condition ID starts with the filter, from the flattened list
    def filter_fn(cond_ids, c_filter):
        flattened = [c for cond_list in cond_ids for c in cond_list]
        return any(c.startswith(c_filter) for c in flattened)
    
    filtered = df[df[filtered_column].apply(lambda x: filter_fn(x, filter_str))]
    return filtered


if __name__ == "__main__":
    main()
    