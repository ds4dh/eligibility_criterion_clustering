import os
import csv
import json
import argparse
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")
try:
    from parse_data import CustomJsonParser
    from parse_utils import ClinicalTrialFilter
    from cluster_utils import ClusterOutput
    from cluster_data import cluster_data_fn
    from generate_utils import compute_scores, generate_llm_response, update_config_filters
except:
    from .parse_data import CustomJsonParser
    from .parse_utils import ClinicalTrialFilter
    from .cluster_utils import ClusterOutput
    from .cluster_data import cluster_data_fn
    from .generate_utils import compute_scores, generate_llm_response, update_config_filters
import pandas as pd
import torchdata.datapipes.iter as dpi
from collections import defaultdict
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate data for clinical trials.")
    parser.add_argument("--embed_model_id", type=str, default="pubmed-bert-sentence", help="EC embedding model ID")
    parser.add_argument("--evaluation_cond_type_filter", type=str, default="C01", help="Main condition filter for evaluated CTs")
    parser.add_argument("--cond_filter_lvl", type=int, default=4, help="Condition filter level")
    parser.add_argument("--itrv_filter_lvl", type=int, default=3, help="Intervention filter level")
    parser.add_argument("--result_dir", type=str, default="./experiments/experiment_3_results", help="Directory for the result csv file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of EC samples evaluated")
    return parser.parse_args()


ARGS = parse_arguments()
EMBED_MODEL_ID = ARGS.embed_model_id
EVALUATION_COND_TYPE_FILTER = ARGS.evaluation_cond_type_filter
COND_FILTER_LVL = ARGS.cond_filter_lvl
ITRV_FILTER_LVL = ARGS.itrv_filter_lvl
RESULT_DIR = ARGS.result_dir
NUM_EVALUATED_SAMPLES = ARGS.num_samples
RESULT_PATH = os.path.join(
    RESULT_DIR,
    "model-%s_type%s_cond-%1i_itrv-%1i.csv" % (EMBED_MODEL_ID, EVALUATION_COND_TYPE_FILTER, COND_FILTER_LVL, ITRV_FILTER_LVL)
)
N_GENERATED_ECS = {"C01": 21, "C04": 30, "C14": 20, "C20": 24}.get(EVALUATION_COND_TYPE_FILTER, 21)  # average ECs per CT values (and default is overall average)
LLM_SYSTEM_PROMPT = "You are an assistant helping to generate eligibility criteria sections for clinical trials."
LLM_USER_PROMPT = """I have a clinical trial that includes the following information:
[CT_DATA_TEXT]
Based on the information above, generate the eligibility criteria section for this clinical trial.
Make sure the generated section includes %i eligibility criteria and has the following format:
Inclusion criteria:
<all inclusion criteria>
Exclusion criteria:
<all exclusion criteria>
""" % N_GENERATED_ECS
NUM_REFORMULATION_TRIALS = 10
REFORMULATION_PROMPT_TEMPLATE = """
You will be provided with an eligibility criterion section from a clinical trial.
Your task is to reformulate the section while preserving its meaning, intent, and key details.
The reformulated text should be written clearly and concisely, with professional and formal language suitable for clinical documentation.
Instructions:
    Make sure the generated section includes %i eligibility criteria.
    The list of criteria can be written in a different order than the original one.
    Ensure that the reformulated section contains the same details and structure as the original but uses different wording.
    Do not introduce, omit, or alter any medical or procedural information from the original text.
    Write the reformulated text as a single cohesive section that maintains the technical tone and clarity expected in clinical trials.
Here is the original eligibility criterion section:

[EC_SECTION]

Provide your reformulated section below:
""" % N_GENERATED_ECS


def main():
    """ Compare clustering to llm methods for generating eligibility criteria
        section in clinical trials
    """
    # Load evaluation dataset
    cfg = config.get_config()
    ct_data, ec_references = get_evaluated_ct_dataset(
        data_path=cfg["FULL_DATA_PATH"],
        cond_type_filters=[EVALUATION_COND_TYPE_FILTER],
    )
    
    # Loop through evaluation dataset to score both methods
    for ct_sample, ec_reference in zip(ct_data, ec_references):
        
        # Check number of rows in csv file
        num_evaluated_cts = get_csv_row_count(RESULT_PATH)
        if num_evaluated_cts >= NUM_EVALUATED_SAMPLES + 1:
            break
        
        # Clustering method for ec-section generation
        update_config_filters(
            ct_data=ct_sample,
            cond_filter_lvl=COND_FILTER_LVL,
            itrv_filter_lvl=ITRV_FILTER_LVL,            
        )  # /!\ this updates "cfg" /!\
        try:
            cluster_output = cluster_data_fn(
                embed_model_id=EMBED_MODEL_ID,
                write_results=False,
                generate_embeddings_hierarchically=True,
            )
            cluster_quality = cluster_output.cluster_metrics["label_free"]["Silhouette score"]
            cluster_ec_section = generate_cluster_ec_section(cluster_output)
        except Exception as e:
            logger.info("Clustering method failed (%s)" % str(e))
            continue
        
        # LLM method for ec-section generation
        ct_path = ct_sample["ct_path"]
        llm_ec_section = generate_llm_ec_section(ct_path)
        
        # Compute and add scores to a csv file
        add_row_to_csv_file(
            ct_path=ct_path,
            reference=ec_reference,
            llm_prediction=llm_ec_section,
            cluster_prediction=cluster_ec_section,
            cluster_quality=cluster_quality,
        )
    
    # Add random performance columns after all result rows were written
    _, ec_references_random = get_evaluated_ct_dataset(
        data_path=cfg["FULL_DATA_PATH"],
        cond_type_filters=[],  # i.e., no filter (to select a random CT)
        random_mode=True,
    )
    add_random_performance(RESULT_PATH, ec_references_random)
    add_ceiling_performance(RESULT_PATH)


def get_csv_row_count(csv_path: str) -> int:
    """ Count number of row entries in csv file
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
    except FileNotFoundError:
        row_count = 0
        
    return row_count


def add_row_to_csv_file(
    ct_path: str,
    reference: str,
    llm_prediction: str,
    cluster_prediction: str,
    cluster_quality: float,
) -> None:
    """ Add results about one evaluated sample for eligibility criterion section
        generation, comparing the clustering method to the llm method
    """
    # Compute performance
    result_dict = {
        "CT Path": ct_path,
        "Cluster Quality": cluster_quality,
        "Cluster EC Section": cluster_prediction,
        "LLM EC Section": llm_prediction,
        "Reference": reference,
    }
    score_dict = compute_scores(
        reference,
        cluster_prediction,
        llm_prediction,
    )
    result_dict.update(score_dict)
    
    # Write to CSV file
    os.makedirs(RESULT_DIR, exist_ok=True)
    file_exists = os.path.isfile(RESULT_PATH)
    with open(RESULT_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:  # headers
            writer.writerow(result_dict.keys())
        writer.writerow(result_dict.values())


def add_random_performance(
    path_to_csv_result_file: str,
    random_references: list[str],
) -> None:
    """ Add random scores (scores using suffled references) to result csv file

    Args:
        path_to_csv_result_file (str): path to the fully generated result file
        random_references (list[str]): list of references
    """
    # Read result file as a pandas dataframe and extracts shuffled references
    result_data = pd.read_csv(path_to_csv_result_file)
    
    # Recompute scores with shuffled references
    random_results = []
    for index, row in tqdm(
        result_data.iterrows(),
        total=len(result_data),
        desc="Computing random scores",
    ):
        reference = random_references[index]
        annoying_bug = True
        n_trials = 0
        
        # Annoying bug almost never happens but is due to a very cryptic error
        # that only occurs with SciBERTScore, for one reference in a thousands,
        # so when this happens, another random reference is simply taken
        while annoying_bug:
            try:
                result_dict = compute_scores(
                    reference=reference,
                    cluster_prediction=row["Cluster EC Section"],
                    llm_prediction=row["LLM EC Section"],
                )
                annoying_bug = False
            except RuntimeError:
                n_trials = n_trials + 1
                new_index = (index - n_trials) % len(random_references)
                reference = random_references[new_index]
        
        random_results.append(result_dict)
    
    # Write shuffled scores in new columns
    for key in random_results[0].keys():
        if "Score" in key:
            result_data[f"Shuffled {key}"] = [result[key] for result in random_results]
    
    # Save the updated dataframe back to CSV
    result_data.to_csv(path_to_csv_result_file, index=False)


def add_ceiling_performance(
    path_to_csv_result_file: str,
) -> None:
    """ Compute ceiling performance for the generation task by comparing
        eligibility sections, reformulated by an LLM, to the original one

    Args:
        result_path (str): path to the original result file
    """
    # Read result file and extract original references
    result_data = pd.read_csv(path_to_csv_result_file)
    references = result_data["Reference"].to_list()
    
    # Loop through each reference to compute ceiling performance
    ceil_results = []
    for reference in tqdm(references, desc="Computing ceiling performance"):
        ceil_dicts = defaultdict(list)
        
        # For each reference, compute match with reformulated versions
        for _ in tqdm(
            range(NUM_REFORMULATION_TRIALS),
            desc="Generating controls",
            leave=False,
        ):
            # Annoying bug almost never happens but is due to a very cryptic error
            # that only occurs with SciBERTScore, for one reference in a thousands,
            # so when this happens, another random reference is simply taken
            annoying_bug = True
            while annoying_bug:
                try:
                    # Reformulate eligibiltiy criterion section
                    user_prompt = REFORMULATION_PROMPT_TEMPLATE.replace("[EC_SECTION]", reference)
                    reformulated = generate_llm_response(
                        user_prompt=user_prompt,
                        system_prompt=LLM_SYSTEM_PROMPT,
                    )
                    
                    # Compute match scores between original and reformulated references
                    ceil_dict = compute_scores(reference, reformulated, reformulated)
                    [ceil_dicts[k].append(v) for k, v in ceil_dict.items()]
                    annoying_bug = False
                
                # Regenerate a new reference when SciBERT-Score bug arises                    
                except RuntimeError:
                    pass
        
        # Record match score average values 
        ceil_avgs = {k: sum(v) / len(v) for k, v in ceil_dicts.items() if "Score" in k}
        ceil_results.append(ceil_avgs)
    
    # Write shuffled scores in new columns
    for key in ceil_results[0].keys():
        if "Score" in key:
            result_data[f"Ceiling {key}"] = [result[key] for result in ceil_results]
    
    # Save the updated dataframe back to CSV
    result_data.to_csv(path_to_csv_result_file, index=False)


def generate_llm_ec_section(ct_path: str) -> str:
    """ Generate the eligibility criterion section by prompting GPT-3.5-turbo with
        clinical trial information

    Args:
        ct_path (str): path to raw clinical trial data file (json)

    Returns:
        str: generated eligibility criterion section
    """
    # Generate a prompt using data from the clinical trial file
    with open(ct_path, "r", encoding="utf-8") as file:
        ct_raw_dict: dict[str, dict|bool] = json.load(file)
    ct_raw_dict["protocolSection"].pop("eligibilityModule")
    ct_raw_dict.pop("resultsSection", None)  # since it wouldn't make sense to
    ct_raw_dict.pop("hasResults", None)      # have results prior to the study
    user_prompt = LLM_USER_PROMPT.replace("[CT_DATA_TEXT]", str(ct_raw_dict))
    
    # Prompt gpt-3.5-turbo
    logger.info("Generating ec section using LLM method")
    return generate_llm_response(
        user_prompt=user_prompt,
        system_prompt=LLM_SYSTEM_PROMPT,
    )


def generate_cluster_ec_section(cluster_output: ClusterOutput) -> str:
    """ Generate the clinical trial section for elibibility criteria based on the
        output of the clustering pipeline

    Args:
        cluster_output (ClusterOutput): formatted clustering pipeline output

    Returns:
        str: generated eligibility criteria section
    """
    logger.info("Generating ec section using cluster method")
    clusters = [c for c in cluster_output.cluster_instances if c.cluster_id != -1]
    clusters.sort(key=lambda cluster: cluster.prevalence, reverse=True)
    selected_ec_texts = [c.ec_list[0].raw_text for c in clusters]
    ec_section = format_ec_section(selected_ec_texts[:N_GENERATED_ECS])
    return ec_section


def format_ec_section(ec_list: list[str]) -> str:
    """ Format a list of eligibility criteria into a text with inclusion criteria
        and exclusion criteria subsections
        
    Args:
        raw_ec_section (str): original eligibility criterion section text
        
    Returns:
        str: transformed section with separate inclusion and exclusion criteria
    """
    inclusion_criteria, exclusion_criteria = [], []
    check_str_in = "inclusion criterion - "
    check_str_ex = "exclusion criterion - "    
    for criterion in ec_list:
        if criterion.lower().startswith(check_str_in):
            inclusion_criteria.append(criterion.split(check_str_in)[-1])
        elif criterion.lower().startswith(check_str_ex):
            exclusion_criteria.append(criterion.split(check_str_ex)[-1])
        else:
            inclusion_criteria.append(criterion)
    
    # Format the output text with the specified sections
    ec_section = "Inclusion criteria:\n" + "\n".join(inclusion_criteria)
    ec_section += "\n\nExclusion criteria:\n" + "\n".join(exclusion_criteria)
    
    return ec_section


def get_evaluated_ct_dataset(
    data_path: str,
    cond_type_filters: list[str],
    random_mode=False,
) -> list[list[dict], list[str]]:
    """ Build a dataset of evaluated clinical trials from raw json files
    
    Args:
        data_path (str): full path to the raw clinical trial data
        
    Returns:
        [
            list[dict]: evaluated clinical trials data
            list[str]: eligibility criteria section references
        ]
    """
    random_str = " (random references only)" if random_mode else ""
    logger.info("Building ec-generation dataset" + random_str)
    files = dpi.FileLister(data_path, recursive=True, masks="*.json")
    shuffled_files = dpi.Shuffler(files)
    json_streams = dpi.FileOpener(shuffled_files, encoding="utf-8")
    parsed_cts = CustomJsonParser(json_streams)
    filtered_cts = ClinicalTrialFilter(parsed_cts)
    
    # Check for existing data from previous runs
    try:
        if not random_mode:
            results_from_last_run = pd.read_csv(RESULT_PATH)
            cts_evaluated_last_runs = results_from_last_run["CT Path"].tolist()
        else:
            cts_evaluated_last_runs = []    
    except FileNotFoundError:
        cts_evaluated_last_runs = []
    
    # Load just enough CT data to complete the previous runs
    input_data, target_data = [], []
    for ct_metadata, ct_ec_section in filtered_cts:
        
        # Filter for condition type (to stratify generation experiment)
        ct_cond_ids = ct_metadata["condition_ids"]
        ct_cond_ids = [c for cc in ct_cond_ids for c in cc]  # flatten
        if len(cond_type_filters) > 0:
            matching_condition_found = False
            for cond_type_filter in cond_type_filters:
                if any([c.startswith(cond_type_filter) for c in ct_cond_ids]):
                    matching_condition_found = True
            if not matching_condition_found:
                continue
        
        # Filter out clinical trials that do not have enough depth in their IDs
        ct_itrv_ids = ct_metadata["condition_ids"]
        ct_itrv_ids = [i for ii in ct_itrv_ids for i in ii]  # flatten
        cond_depths = [len(cond_id.split(".")) for cond_id in ct_cond_ids]
        itrv_depths = [len(itrv_id.split(".")) for itrv_id in ct_itrv_ids]
        if all([d < COND_FILTER_LVL for d in cond_depths])\
        or all([d < ITRV_FILTER_LVL for d in itrv_depths]):
            continue
        
        # Filter out clinical trials with phase different from 1, 2, or 3.
        ct_phases = ct_metadata["phases"]
        chosen_phases = ["phase1", "phase2", "phase3"]
        if all([p not in chosen_phases for p in ct_phases]):
            continue
        
        # Check is run is finished or if ct has already been run previously
        if len(input_data) >= NUM_EVALUATED_SAMPLES * 10:
            break
        if len(cts_evaluated_last_runs) + len(input_data) >= NUM_EVALUATED_SAMPLES * 10:
            break
        if ct_metadata["ct_path"] in cts_evaluated_last_runs:
            continue
        
        # Include this ct in the dataset
        input_data.append(ct_metadata)
        target_data.append(ct_ec_section)
    
    return input_data, target_data


if __name__ == "__main__":
    main()