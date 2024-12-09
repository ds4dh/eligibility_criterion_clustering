import os
import yaml
import logging
import threading
from datetime import datetime

configuration = None  # global configuration dictionary
config_lock = threading.Lock()  # lock for thread-safe configuration updates


def load_default_config(default_path: str="config.yaml"):
    """ Load default configuration file to memory
    """
    global configuration
    with open(default_path, "r") as file:
        with config_lock:
            configuration = yaml.safe_load(file)


def update_config(request_data: dict):
    """ Update the in-memory configuration with a dictionary of updates
    """
    global configuration
    with config_lock:
        for key, value in request_data.items():
            if isinstance(configuration.get(key), dict) and isinstance(value, dict):
                configuration[key].update(value)
            else:
                configuration[key] = value


def get_config(default_path: str="config.yaml"):
    """ Get the current in-memory configuration (or default one if non-existent)
    """
    global configuration
    if configuration is None:
        load_default_config(default_path)
    return align_config(configuration)


def align_config(cfg):
    """ Make sure eligibility criteria are not filtered when running environment
        is ctxai, then register and create all required paths and directories
    """
    # Helper logging function
    def log_warning(field, value, culprit_field, culprit_value):
        logger.warning(
            "Config field %s was changed to %s because %s is %s" \
            % (field, value, culprit_field, culprit_value)
        )
    
    # Check in which environment eligibility criteria clustering is run
    match cfg["ENVIRONMENT"]:
        case "ctgov":
            base_dir = "data_ctgov"
        case "ctxai_dev":
            base_dir = os.path.join("data_dev", "upload")
        case "ctxai_prod":
            base_dir = os.path.join("data_prod", "upload")
        case _:
            raise ValueError("Invalid ENVIRONMENT config variable.")
        
    # Check filters given data format
    logger = CTxAILogger("INFO")
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
    
    # Generate a common directory for all outputs of the pipeline
    if cfg["SELECT_USER_ID_AND_PROJECT_ID_AUTOMATICALLY"]:
        cond_itrv_str = "-".join(cfg["CHOSEN_COND_IDS"] + cfg["CHOSEN_ITRV_IDS"])
        cfg["USER_ID"] = "%s-%s" % (cfg["ENVIRONMENT"], cond_itrv_str)
        cfg["PROJECT_ID"] = "cond-lvl-%s_itrv-lvl-%s_cluster-%s-%s_plot-%s-%s" % (
            cfg["CHOSEN_COND_LVL"], cfg["CHOSEN_ITRV_LVL"],
            cfg["CLUSTER_DIM_RED_ALGO"], cfg["CLUSTER_RED_DIM"],
            cfg["PLOT_DIM_RED_ALGO"], cfg["PLOT_RED_DIM"],
        )
    output_dir = os.path.join(base_dir, cfg["USER_ID"], cfg["PROJECT_ID"])
    if cfg["USER_FILTERING"] is not None:
        output_dir = os.path.join(output_dir, cfg["USER_FILTERING"])
    
    # Create and register all required paths and directories
    cfg["FULL_DATA_PATH"] = os.path.join(base_dir, cfg["DATA_PATH"])
    cfg["PROCESSED_DIR"] = os.path.join(output_dir, "processed")
    cfg["RESULT_DIR"] = os.path.join(output_dir, "results")
    
    # Special case for ctgov environment, where pre-processed data is re-used
    if cfg["ENVIRONMENT"] == "ctgov":
        cfg["PREPROCESSED_DIR"] = base_dir  # cfg["FULL_DATA_PATH"]
    else:
        cfg["PREPROCESSED_DIR"] = cfg["PROCESSED_DIR"]
        
    return cfg


class CTxAILogger:
    """ Modified from BERTopic -> https://maartengr.github.io/BERTopic/index.html
    """
    def __init__(self, level, log_path="logs/app.log"):
        self.logger = logging.getLogger("CTxAI")
        self.set_level(level)
        self.log_path = self._set_log_file(log_path)
        self._add_handlers()
        self.logger.propagate = False
        
    def _set_log_file(self, log_path=None, timed=False):
        # Check log dir exists
        if log_path is None: return None
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timed log path if required
        if timed:
            log_name_with_ext = os.path.basename(log_path)
            log_name, log_ext = os.path.splitext(log_name_with_ext)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            log_path = os.path.join(log_dir, f"{log_name}_{timestamp}{log_ext}")
        
        return log_path
    
    def info(self, message):
        self.logger.info(f"{message}")
        
    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")
        
    def error(self, message):
        self.logger.error(f"ERROR: {message}")
        
    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)
            
    def _add_handlers(self):
        # StreamHandler for console logging
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
        self._add_handler_if_unique(sh)
        
        # Optional FileHandler for file logging
        if self.log_path is not None:
            fh = logging.FileHandler(self.log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
            self._add_handler_if_unique(fh)
    
    def _add_handler_if_unique(self, new_handler: logging.Handler):
        for handler in self.logger.handlers:
            if handler.__class__ == new_handler.__class__\
            and handler.formatter._fmt == new_handler.formatter._fmt:
                return  # avoid adding duplicate handler
        self.logger.addHandler(new_handler)