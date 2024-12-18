import os
import yaml
import logging
import warnings
import optuna
from functools import partial
from contextlib import contextmanager
from tqdm import tqdm as original_tqdm


def load_default_config(default_path: str="config.yaml") -> dict:
    """ Load default configuration file to memory
    """
    with open(default_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def update_config(cfg: dict, request_data: dict):
    """ Update the in-memory configuration with a dictionary of updates
    """
    # Update configuration with new data
    for key, value in request_data.items():
        if isinstance(cfg.get(key), dict) and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    
    # Re-align updated config in case anything crucial is chaged
    cfg = align_config(cfg)
    
    return cfg


def align_config(cfg):
    """ Make sure eligibility criteria are not filtered when running environment
        is ctxai, then register and create all required paths and directories
    """
    # Helper logging function
    def log_warning(field, value, culprit_field, culprit_value):
        warnings.warn(
            "Config field %s was changed to %s because %s is %s" \
            % (field, value, culprit_field, culprit_value),
            category=UserWarning,
        )
    
    # Check in which environment eligibility criteria clustering is run
    match cfg["ENVIRONMENT"]:
        case "ctgov":
            cfg["BASE_DIR"] = "data_ctgov"
        case "ctxai_dev":
            cfg["BASE_DIR"] = os.path.join("data_dev", "upload")
        case "ctxai_prod":
            cfg["BASE_DIR"] = os.path.join("data_prod", "upload")
        case _:
            raise ValueError("Invalid ENVIRONMENT config variable.")
        
    # Check filters given data format
    if "ctxai" in cfg["ENVIRONMENT"]:
        
        # Make sure USER_ID and PROJECT_ID are not overwritten by script
        field = "SELECT_USER_ID_AND_PROJECT_ID_AUTOMATICALLY"
        if cfg[field] != False:
            cfg[field] = False
            log_warning(field, False, "ENVIRONMENT", "ctxai")
        
        # Make sure loading from cache is disabled for ctxai environment
        for field in [
            "LOAD_PARSED_DATA",
            "LOAD_EMBEDDINGS",
            "LOAD_BERTOPIC_RESULTS",
        ]:
            if cfg[field] != False:
                cfg[field] = False
                log_warning(field, False, "ENVIRONMENT", "ctxai")
        
        # Make sure no criteria filtering is applied for ctxai environment, since
        # ctxai data is already filtered by the upstream user
        for field in [
            "CHOSEN_STATUSES",
            "CHOSEN_CRITERIA",
            "CHOSEN_PHASES",
            "CHOSEN_COND_IDS",
            "CHOSEN_ITRV_IDS",
        ]:
            if cfg[field] != []:
                cfg[field] = []
                log_warning(field, [], "ENVIRONMENT", "ctxai")
        
        # Make sure no criteria filtering is applied for ctxai environment, since
        # ctxai data is already filtered by the upstream user
        for field in [
            "CHOSEN_COND_LVL",
            "CHOSEN_ITRV_LVL",
            "STATUS_MAP",
        ]:
            if cfg[field] is not None:
                cfg[field] = None
                log_warning(field, None, "ENVIRONMENT", "ctxai")
    
    # Generate user and project id based on configuration, if required
    if cfg["SELECT_USER_ID_AND_PROJECT_ID_AUTOMATICALLY"]:
        cond_itrv_str = "-".join(cfg["CHOSEN_COND_IDS"] + cfg["CHOSEN_ITRV_IDS"])
        cfg["USER_ID"] = "%s-%s" % (cfg["ENVIRONMENT"], cond_itrv_str)
        cfg["PROJECT_ID"] = "cond-lvl-%s_itrv-lvl-%s_cluster-%s-%s_plot-%s-%s" % (
            cfg["CHOSEN_COND_LVL"], cfg["CHOSEN_ITRV_LVL"],
            cfg["CLUSTER_DIM_RED_ALGO"], cfg["CLUSTER_RED_DIM"],
            cfg["PLOT_DIM_RED_ALGO"], cfg["PLOT_RED_DIM"],
        )
    
    # Determine user and output directory, where all outputs will be saved
    cfg["USER_DIR"] = os.path.join(cfg["BASE_DIR"], cfg["USER_ID"])
    cfg["PROJECT_DIR"] = os.path.join(cfg["USER_DIR"], cfg["PROJECT_ID"])
    if cfg["USER_FILTERING"] is not None:
        cfg["PROJECT_DIR"] = os.path.join(cfg["PROJECT_DIR"], cfg["USER_FILTERING"])
        
    # Create and register all required paths and directories
    cfg["FULL_DATA_PATH"] = os.path.join(cfg["BASE_DIR"], cfg["DATA_PATH"])
    cfg["PROCESSED_DIR"] = os.path.join(cfg["PROJECT_DIR"], "processed")
    cfg["RESULT_DIR"] = os.path.join(cfg["PROJECT_DIR"], "results")
    
    # Special case for ctgov environment, where pre-processed may be re-used
    if cfg["ENVIRONMENT"] == "ctgov":
        cfg["PREPROCESSED_DIR"] = cfg["BASE_DIR"]  # cfg["FULL_DATA_PATH"]
    else:
        cfg["PREPROCESSED_DIR"] = cfg["PROCESSED_DIR"]
        
    return cfg


class CTxAILogger:
    """ Modified from BERTopic -> https://maartengr.github.io/BERTopic/index.html
    """
    def __init__(
        self,
        session_id: str,
        level: str="INFO",
        log_directory: str="logs",
        max_log_file_lines: int=1000,
    ):
        self.session_id = session_id
        self.level = level
        self.max_log_file_lines = max_log_file_lines
        
        # Compute a user-specific log path
        os.makedirs(log_directory, exist_ok=True)
        self.log_path = os.path.join(log_directory, f"app_{session_id}.log")
        
        # Create a session-specific logger instance
        self.logger = logging.getLogger(f"CTxAI_{session_id}")
        self.logger.setLevel(level)
        
        # Add a stream handler (and avoid duplicating it)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s - CTxAI - %(message)s"))
            self.logger.addHandler(sh)
        
        # Add a file handler (and avoid duplicating it)
        if not os.path.exists(self.log_path):
            
            # Remove any existing FileHandler first
            for handler in self.logger.handlers[:]:  # [:] to avoid modifying the list
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
                    handler.close()
                    
            fh = logging.FileHandler(self.log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s - CTxAI - %(message)s"))
            self.logger.addHandler(fh)
            
    def info(self, message):
        self._truncate_log_file()
        self.logger.info(message)
        
    def warning(self, message):
        self._truncate_log_file()
        self.logger.warning(message)
        
    def error(self, message):
        self._truncate_log_file()
        self.logger.error(message)
        
    def _truncate_log_file(self):
        """ Truncate the log file to a fixed number of lines by deleting the
            first lines of the file
        """
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "r+") as log_file:
            lines = log_file.readlines()
            if len(lines) > self.max_log_file_lines:
                log_file.seek(0)
                log_file.writelines(lines[-self.max_log_file_lines:])
                log_file.truncate()
                

class CustomTqdm(original_tqdm):
    """ Custom tqdm class with suffix and prefix and sending output to a logger
    """
    def __init__(
        self,
        *args,
        logger: CTxAILogger=None,
        prefix: str="",
        suffix: str="",
        **kwargs,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.logger = logger
        super().__init__(*args, **kwargs)
    
    def set_description(self, desc=None, refresh=True):
        if desc is not None:
            desc = f"{self.prefix}{desc}{self.suffix}".strip()
        super().set_description(desc, refresh)
        
    def display(self, msg=None, pos=None):
        if self.logger:
            self.logger.info(msg or self.__str__())
        else:
            super().display(msg, pos)
            
            
@contextmanager
def optuna_with_custom_tqdm(
    logger: CTxAILogger=None,
    prefix: str="",
    suffix: str="",
    **kwargs,
):
    """ Context manager to temporarily replace tqdm with CustomTqdm
    """
    # Create a custom tqdm class, with specified parameters
    custom_tqdm = partial(
        CustomTqdm,
        logger=logger,
        prefix=prefix,
        suffix=suffix,
        **kwargs,
    )
    
    # Context manager logic
    original_tqdm_module = optuna.progress_bar.tqdm
    try:
        optuna.progress_bar.tqdm = custom_tqdm
        yield
        
    # Restore original tqdm when exiting context
    finally:
        optuna.progress_bar.tqdm = original_tqdm_module 
        