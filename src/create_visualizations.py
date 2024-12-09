import os
import shutil
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")
import torchdata.datapipes.iter as dpi
from typing import Any
try:
    from config import update_config
    from cluster_data import cluster_data_fn
    from generate_utils import extract_ids_at_lvl
    from parse_utils import (
        ClinicalTrialFilter, CustomJsonParser, CustomNCTIdParser, CustomCTDictNamer,
    )
except:
    from .config import update_config
    from .cluster_data import cluster_data_fn
    from .generate_utils import extract_ids_at_lvl
    from .parse_utils import (
        ClinicalTrialFilter, CustomJsonParser, CustomNCTIdParser, CustomCTDictNamer,
    )


EMBED_MODEL_ID = "pubmed-bert-sentence"
COND_TYPE_FILTER_SETS = [["C01"], ["C04"], ["C14"], ["C20"]]
COND_FILTER_LVL = 4
ITRV_FILTER_LVL = 3
DATA_DIR = "data_ctgov"
CT_FILE_DIR = os.path.join(DATA_DIR, "raw_files/NCT0030xxxx")
CT_FILE_PATHS = [os.path.join(CT_FILE_DIR, p) for p in os.listdir(CT_FILE_DIR)]
# CT_FILE_PATHS = [
#     # C01 clinical trials
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00303550.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00307489.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00308048.json",
    
#     # C04 clinical trials
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00301847.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00306969.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00309556.json",
    
#     # C14 clinical trials
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00304226.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00306735.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00307307.json",
    
#     # C20 clinical trials
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00304538.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00306969.json",
#     "data_ctgov/raw_files/NCT0030xxxx/NCT00308620.json",
# ]


def main():
    """ Generate a few cluster output visualization for clinical trials coming
        from different condition type filters
    """
    for cond_type_filter_set in COND_TYPE_FILTER_SETS:
        create_visualizations_from_ct_paths_or_nct_ids(
            ct_file_paths=CT_FILE_PATHS,
            n_examples_to_generate=1,
            cond_type_filter_set=cond_type_filter_set,
        )


def create_visualizations_from_ct_paths_or_nct_ids(
    n_examples_to_generate: int,
    manuscript_mode: bool=False,
    ct_dicts: list[dict[str, Any]]|None=None,
    ct_nct_ids: list[str]|None=None,
    ct_file_paths: list[str]|None=None,
    cond_type_filter_set: list[str]=[],
) -> list[dict[str, str]]:
    """ Generate a cluster output for eligibility criteria coming from similar
        clinical trials, for a few examples of clinical trials of a given set of
        condition filters
    """
    # Figure out what to process
    assert sum(x is not None for x in [ct_dicts, ct_nct_ids, ct_file_paths]) == 1
    if ct_dicts is not None:
        named_ct_dicts = CustomCTDictNamer(ct_dicts)
    elif ct_nct_ids is not None:
        named_ct_dicts = CustomNCTIdParser(ct_nct_ids)
    elif ct_file_paths is not None:
        json_streams = dpi.FileOpener(ct_file_paths, encoding="utf-8")
        named_ct_dicts = CustomJsonParser(json_streams)
    filtered_ct_dicts = ClinicalTrialFilter(named_ct_dicts, for_inference=True)
    
    # Loop through clinical trial parsed csv files
    n_generated_examples = 0
    visualization_outputs = []
    for ct_data, _ in filtered_ct_dicts:  # "_" is eligibility section (not used here)
        
        # Try to create a visualization and skip the sample if it failed
        try:
            visualization_output = create_visualization(ct_data, cond_type_filter_set)
        except RuntimeError as visualization_error:
            if manuscript_mode:
                logger.info("This clinical trial was not eligible, skipping")
                continue
            else:
                raise(visualization_error)
        visualization_outputs.append(visualization_output)
        
        # Check if enough examples were generated
        logger.info(f"Generated visualization at {visualization_output['html']}")
        n_generated_examples += 1
        if n_generated_examples >= n_examples_to_generate:
            break
    
    return visualization_outputs


def create_visualization(
    ct_data: dict[str, Any],
    cond_type_filter_set: list[str]=[],
) -> dict[str, str]|None:
    """ Generate visualization from the yielded output of ClinicalTrialFilter
    """
    # Extract phase(s), condition(s), and intervention(s) and check there are enough
    ct_path, ct_phases, cond_ids, itrv_ids = extract_ct_info_for_visualization(ct_data)
    if len(ct_phases) == 0:
        raise RuntimeError("No phase identified in the CTs data")
    elif len(cond_ids) == 0 and len(itrv_ids) == 0:
        raise RuntimeError("Neither condition or intervention ID identified in the CTs data")
    
    # If any, check for the presence of the wanted condition type(s)
    if len(cond_type_filter_set) > 0:
        if not any(f in c for c in cond_ids for f in cond_type_filter_set):
            raise RuntimeError("No condition ID matches the filtered condition type")
    
    # Try to generate a cluster visualization
    visualization_output = create_visualization_from_ct_info(
        ct_path=ct_path, ct_phases=ct_phases,
        cond_ids=cond_ids, itrv_ids=itrv_ids,
    )
    
    return visualization_output


def extract_ct_info_for_visualization(
    ct_data: dict[str, Any],
) -> tuple[str, list[str], list[str], list[str]]:
    """ Extract relevant information from a clinical trial to be used by the
        clustering pipeline
    """
    ct_path = ct_data["ct_path"]
    ct_phases = ct_data["phases"]
    cond_ids = extract_ids_at_lvl(ct_data["condition_ids"], COND_FILTER_LVL)
    itrv_ids = extract_ids_at_lvl(ct_data["intervention_ids"], ITRV_FILTER_LVL)
    
    return ct_path, ct_phases, cond_ids, itrv_ids
    

def create_visualization_from_ct_info(
    ct_phases: list[str],
    cond_ids: list[str],
    itrv_ids: list[str],
    ct_path: str|None=None,
    cond_type_filter_set: list[str]=[],
) -> dict[str, str]|None:
    """ Generate a cluster visualization from a set of phases, conditions, and
        interventions, then return the path to the HTML visualization files
    """
    # Identify where data will be written
    user_id = "-".join(["examples_with_CT"] + cond_type_filter_set)
    project_id = ct_path.split("/")[-1].replace(".json", "")
    
    # Update configuration with CT phase(s), condition(s), and intervention(s)
    to_update = {
        "CHOSEN_PHASES": ct_phases,
        "CHOSEN_COND_IDS": cond_ids,
        "CHOSEN_ITRV_IDS": itrv_ids,
        "CHOSEN_COND_LVL": COND_FILTER_LVL,
        "CHOSEN_ITRV_LVL": ITRV_FILTER_LVL,
        "DO_EVALUATE_CLUSTERING": True,
        "MAX_EC_SAMPLES": 50_000,
        "MIN_EC_SAMPLES": 2_500,
        "N_OPTUNA_TRIALS": 25,
        "USER_ID": user_id,
        "PROJECT_ID": project_id,
        # "CLUSTER_REPRESENTATION_MODEL": "gpt",
        "SELECT_USER_ID_AND_PROJECT_ID_AUTOMATICALLY": False,
    }
    if ct_path is not None:
        to_update.update({"ADDITIONAL_NEGATIVE_FILTER": {"ct path": ct_path}})        
    update_config(request_data=to_update)  # /!\ this updates "cfg" object /!\
    
    # Cluster eligibility criteria similar to the current CT (using filters)
    try:
        cluster_output = cluster_data_fn(
            embed_model_id=EMBED_MODEL_ID,
            write_results=True,
            hierarchical_ec_scraping=True,
        )
    
    # Clustering may fail if there are not enough similiar criteria
    except RuntimeError as cluster_algorithm_error:
        shutil.rmtree(os.path.join(DATA_DIR, user_id, project_id))
        raise(cluster_algorithm_error)
    
    # Write information about this cluster run to a log file
    html_file_path = cluster_output.visualization_paths["all"]["html"]
    log_file_path = html_file_path.replace("html", "log")
    n_ecs = cluster_output.n_samples
    log_entry = (
        f"CT path: {ct_path}\n"
        f"Number of eligibility criteria from similar CTs: {n_ecs}\n"
        f"CT phases: {', '.join(ct_phases)}\n"
        f"CT condition IDs: {', '.join(cond_ids)}\n"
        f"CT intervention IDs: {', '.join(itrv_ids)}"
    )
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_entry)
    
    return {"html": html_file_path, "log": log_file_path}


if __name__ == "__main__":
    main()
    