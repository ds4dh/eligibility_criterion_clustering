import os
import csv
import json
import time
import argparse
import config
logger = config.CTxAILogger("INFO")
import pandas as pd
import torchdata.datapipes.iter as dpi
import openai
from openai import OpenAI
from parse_data import CustomJsonParser
from parse_utils import ClinicalTrialFilter
from cluster_utils import ClusterOutput
from cluster_data import cluster_data_fn
from generate_utils import compute_scores
from config import update_config
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate data for clinical trials.")
    parser.add_argument("--embed_model_id", type=str, default="pubmed-bert-sentence", help="EC embedding model ID")
    parser.add_argument("--evaluation_cond_type_filter", type=str, default="C01", help="Main condition filter for evaluated CTs")
    parser.add_argument("--cond_filter_lvl", type=int, default=2, help="Condition filter level")
    parser.add_argument("--itrv_filter_lvl", type=int, default=1, help="Intervention filter level")
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
RANDOM_COND_TYPE_FILTERS = [
    "C05", "C06", "C07", "C08", "C09", "C10",
    "C11", "C12", "C15", "C16", "C17", "C18", "C19",
]  # anything that is not in ["C01", "C04", "C14", "C20"] and that is below "C20"
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
FAILED_CLUSTERING_TEXT = "Clustering did not converge"


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
        
        # LLM method for ec-section generation
        ct_path = ct_sample["ct_path"]
        llm_ec_section = generate_llm_ec_section(ct_path)
        
        # Clustering method for ec-section generation
        update_config_filters(ct_sample)  # /!\ this updates "cfg" /!\
        try:
            cluster_output = cluster_data_fn(EMBED_MODEL_ID, write_results=False)
            cluster_quality = cluster_output.cluster_metrics["label_free"]["Silhouette score"]
            cluster_ec_section = generate_cluster_ec_section(cluster_output)
        except Exception as e:
            logger.info("Clustering method failed (%s)" % str(e))
            cluster_ec_section = FAILED_CLUSTERING_TEXT  # default cluster ec-section
            cluster_quality = -1.0  # minimum value
        
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
        cond_type_filters=RANDOM_COND_TYPE_FILTERS,
        random_mode=True,
    )
    add_random_performance(RESULT_PATH, ec_references_random)


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
    """
    # Read result file as a pandas dataframe and extracts shuffled references
    df = pd.read_csv(path_to_csv_result_file)
    # df["Cluster EC Section"] = df["Cluster EC Section"].fillna(FAILED_CLUSTERING_TEXT)  # NaN values
    # references = df["Reference"].tolist()
    # random_references = references[:]
    # random.shuffle(random_references)
    
    # Recompute scores with shuffled references
    random_results = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Computing random scores"):
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
            df[f"Shuffled {key}"] = [result[key] for result in random_results]
    
    # Save the updated dataframe back to CSV
    df.to_csv(path_to_csv_result_file, index=False)


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
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            client = OpenAI(api_key=get_openai_api_key())
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            # Extract the generated EC section from the API response
            ec_section: str = response.choices[0].message.content
            
            return ec_section.strip()
        
        # Expected error handling
        except openai.OpenAIError as e:
            logger.error("Openai error occured (%s), retrying" % str(e))
            user_prompt = user_prompt[int(len(user_prompt) * 0.9)]
        
        # Exponential back-off
        time.sleep(2 ** attempt)
        
    return "Failed to generate EC section after multiple attempts."


def get_openai_api_key():
    """ Retrieve the OpenAI API key from the configuration file
    
    Returns:
        str: the OpenAI API key
    """
    cfg = config.get_config()
    api_path = os.path.join(cfg["CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY"])
    try:
        with open(api_path, "r", encoding="utf-8") as f: return f.read()
    except:
        raise FileNotFoundError(" ".join([
            "To use CLUSTER_REPRESENTATION_MODEL = gpt,",
            "you must have an api-key file at the path defined in the",
            "config under CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY",
        ]))


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


def update_config_filters(ct_data: dict) -> None:
    """ Update running configuration with phase, condition and intervention
        filters for later clustering 
        
    Args:
        ct_data (dict): clinical trial data
    """
    # Identify phase(s), condition(s), and intervention(s) of the clinical trial
    cond_ids = extract_ids_at_lvl(ct_data["condition_ids"], COND_FILTER_LVL)
    itrv_ids = extract_ids_at_lvl(ct_data["intervention_ids"], ITRV_FILTER_LVL)
    to_update = {
        "CHOSEN_PHASES": ct_data["phases"],
        "CHOSEN_COND_IDS": cond_ids,
        "CHOSEN_ITRV_IDS": itrv_ids,
    }
    logger.info("Evaluating ec-generation for ct with: %s" % to_update)
    
    # Make sure that evaluated clinical trial is not considered for clustering 
    to_update.update({
        "DO_EVALUATE_CLUSTERING": True,
        "ADDITIONAL_NEGATIVE_FILTER": {"ct path": ct_data["ct_path"]},
        "MAX_ELIGIBILITY_CRITERIA_SAMPLES": 50_000,
        "N_OPTUNA_TRIALS": 25,  # not harmful in 99% of cases, and else too slow
    })
    
    # Update globally shared configuration with current information
    update_config(request_data=to_update)


def extract_ids_at_lvl(ids: str, lvl: int) -> list[str]:
    """ Extract condition or intervention id(s) up to a certain MeSH tree level

    Args:
        ids (str): raw condition or intervention id(s)
        lvl (int): level up to which id(s) are considered

    Returns:
        list[str]: list of unique ids up to the required level
    """
    flat_ids = [id for id_sublist in ids for id in id_sublist]
    extracted_ids = [".".join(id.split(".")[:lvl]) for id in flat_ids]
    return list(set(extracted_ids))


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
    logger.info("Building ec-generation dataset")
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
        matching_condition_found = False
        for cond_type_filter in cond_type_filters:
            if any([c.startswith(cond_type_filter) for c in ct_cond_ids]):
                matching_condition_found = True
        if not matching_condition_found: continue
                
        # Check is run is finished or if ct has already been run previously
        if len(input_data) >= NUM_EVALUATED_SAMPLES:
            break
        if len(cts_evaluated_last_runs) + len(input_data) >= NUM_EVALUATED_SAMPLES:
            break
        if ct_metadata["ct_path"] in cts_evaluated_last_runs:
            continue
        
        # Include this ct in the dataset
        input_data.append(ct_metadata)
        target_data.append(ct_ec_section)
    
    return input_data, target_data


if __name__ == "__main__":
    main()