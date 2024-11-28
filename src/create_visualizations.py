import os
import shutil
try:
    import config
except:
    from . import config
logger = config.CTxAILogger("INFO")
import torchdata.datapipes.iter as dpi
try:
    from cluster_data import cluster_data_fn
    from generate_data import update_config_filters
    from parse_data import CustomJsonParser
    from parse_utils import ClinicalTrialFilter
except:
    from .cluster_data import cluster_data_fn
    from .generate_data import update_config_filters
    from .parse_data import CustomJsonParser
    from .parse_utils import ClinicalTrialFilter


EMBED_MODEL_ID = "pubmed-bert-sentence"
COND_TYPE_FILTER_SETS = [["C01"], ["C04"], ["C14"], ["C20"]]
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
        generate_example_visualizations(
            n_examples_to_generate=1,
            cond_type_filter_set=cond_type_filter_set,
        )
        
        
def generate_example_visualizations(
    n_examples_to_generate:int,
    cond_type_filter_set: list[str]|None=None,
) -> None:
    """ Generate a cluster output for eligibility criteria coming from similar
        clinical trials, for a few examples of clinical trials of a given set of
        condition filters
    """
    # Loop through clinical trial parsed csv files
    n_generated_examples = 0
    json_streams = dpi.FileOpener(CT_FILE_PATHS, encoding="utf-8")
    parsed_cts = CustomJsonParser(json_streams)
    filtered_cts = ClinicalTrialFilter(parsed_cts)
    for ct_sample, _ in filtered_cts:
        
        # Extract information and set save folders
        ct_path = ct_sample["ct_path"]
        user_id = "-".join(["examples_with_CT"] + cond_type_filter_set)
        project_id = ct_path.split("/")[-1].replace(".json", "")
        
        # Check for at least one phase, condition and intervention ids in the CT
        ct_phases = ct_sample["phases"]
        cond_ids = [c for conds in ct_sample["condition_ids"] for c in conds]
        itrv_ids = [i for itrvs in ct_sample["intervention_ids"] for i in itrvs]
        if len(ct_phases) == 0 or len(cond_ids) == 0 or len(itrv_ids) == 0:
            continue
        
        # If any, check for the presence of the wanted condition type(s)
        if cond_type_filter_set is not None:
            if not any(f in c for c in cond_ids for f in cond_type_filter_set):
                continue
        
        # Set phase, condition(s), and intervention(s) of the CT as filters
        update_config_filters(
            ct_data=ct_sample, MIN_EC_SAMPLES=2_500,
            SELECT_USER_ID_AND_PROJECT_ID_AUTOMATICALLY=False,
            USER_ID=user_id, PROJECT_ID=project_id,
            N_OPTUNA_TRIALS=100, CLUSTER_REPRESENTATION_MODEL="gpt",
        )  # /!\ this updates "cfg" object /!\
        
        # Cluster eligibility criteria similar to the current CT (using filters)
        try:
            cluster_output = cluster_data_fn(
                embed_model_id=EMBED_MODEL_ID,
                write_results=True,
                hierarchical_ec_scraping=True,
            )
        except RuntimeError:
            logger.info("Not enough ECs identified for this CT, skipping")
            shutil.rmtree(os.path.join(DATA_DIR, user_id, project_id))
            continue
                
        # Write information about this cluster run to a log file
        html_path = cluster_output.visualization_paths["all"]["html"]
        log_file_path = html_path.replace("html", "log")
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
        
        # Check if enough examples were generated
        logger.info(f"Generated visualization at {html_path}")
        n_generated_examples += 1
        if n_generated_examples >= n_examples_to_generate:
            break
            
            
if __name__ == "__main__":
    main()
    