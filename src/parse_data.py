# Utils
import os
import csv
import torchdata.datapipes.iter as dpi
from flask import g
from tqdm import tqdm
from torchdata.dataloader2 import (
    DataLoader2,
    InProcessReadingService,
    MultiProcessingReadingService,
)
try:
    from cluster_utils import set_seeds
    from parse_utils import (
        ClinicalTrialFilter,
        CriteriaParser,
        CriteriaCSVWriter,
        CustomXLSXLineReader,
        CustomJsonParser,
    )
except:
    from .cluster_utils import set_seeds
    from .parse_utils import (
        ClinicalTrialFilter,
        CriteriaParser,
        CriteriaCSVWriter,
        CustomXLSXLineReader,
        CustomJsonParser,
    )


def main():
    """ Main script (if not run from a web-service)
    """
    parse_data_fn()


def parse_data_fn() -> None:
    """ Parse all CT files into lists of inclusion and exclusion criteria
    """
    # Ensure reproducibility (required here?)
    set_seeds(g.cfg["RANDOM_STATE"])
    
    # Load parsed data from previous run
    if g.cfg["LOAD_PARSED_DATA"]:
        g.logger.info("Eligibility criteria already parsed, skipping this step")
    
    # Parse data using torchdata pipeline
    else:
        g.logger.info("Parsing criteria from raw clinical trial texts")
        
        # Initialize output file with data headers
        csv_path = os.path.join(g.cfg["PREPROCESSED_DIR"], "parsed_criteria.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(g.cfg["PARSED_DATA_HEADERS"])
            
        # Initialize data processors
        too_many_workers = max(os.cpu_count() - 4, os.cpu_count() // 4)
        num_parse_workers = min(g.cfg["NUM_PARSE_WORKERS"], too_many_workers)
        ds = get_dataset(g.cfg["FULL_DATA_PATH"], g.cfg["ENVIRONMENT"])
        if num_parse_workers == 0:
            rs = InProcessReadingService()
        else:
            rs = MultiProcessingReadingService(num_workers=num_parse_workers)
        dl = DataLoader2(ds, reading_service=rs)
        
        # Write parsed criteria to the output file
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                for data in tqdm(dl, desc="Clinical trials processed so far"):
                    writer = csv.writer(f)
                    writer.writerows(data)
        except Exception as e:
            if "Can not request next item" in str(e):
                g.logger.warning("Failed to close processes properly, still continuing")
            else:
                raise
            
        # Close data pipeline
        dl.shutdown()
        g.logger.info("All criteria have been parsed")
        
    
def get_dataset(data_path: str, environment: str) -> dpi.IterDataPipe:
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    # Load correct files from a directory
    if os.path.isdir(data_path):
        masks = "*.%s" % ("json" if environment == "ctgov" else "xlsx")
        files = dpi.FileLister(data_path, recursive=True, masks=masks)
        
    # Load correct file from a file path
    elif os.path.isfile(data_path):
        files = dpi.IterableWrapper([data_path])
    
    # Handle exception
    else:
        raise FileNotFoundError(f"{data_path} is neither a file nor a directory")
    
    # Load data inside each file
    if environment == "ctgov":
        jsons = dpi.FileOpener(files, encoding="utf-8")
        jsons = dpi.ShardingFilter(jsons)
        dicts = CustomJsonParser(jsons)
        raw_samples = ClinicalTrialFilter(dicts)
    elif "ctxai" in environment:
        raw_samples = CustomXLSXLineReader(files)
        raw_samples = dpi.ShardingFilter(raw_samples)
    else:
        raise ValueError("Incorrect ENVIRONMENT field in config.")
        
    # Parse criteria
    parsed_samples = CriteriaParser(raw_samples)
    written = CriteriaCSVWriter(parsed_samples)
    return written


if __name__ == "__main__":
    main()
    