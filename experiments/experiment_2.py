import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
import time
import json
import signal
import subprocess
from src.config_utils import CTxAILogger, load_default_config, update_config
from flask import Flask, g


def main():
    app = Flask(__name__)
    with app.app_context():
        g.session_id = "experiment_2"
        g.cfg = load_default_config()
        g.cfg = update_config(cfg=g.cfg, request_data={"SESSION_ID": g.session_id})
        g.logger = CTxAILogger(level="INFO", session_id=g.session_id)
        experiment_2_fn()
    
    
def experiment_2_fn():
    """ Run the clustering pipeline for many different experimental conditions
    """
    # Start the WSGI server
    wsgi_process = run_wsgi_server()
    
    # Give the server some time to start
    time.sleep(20)
    
    try:
        # Define your sequence of environments and data paths
        tasks = [
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_PHASES": ["phase1"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_PHASES": ["phase2"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_PHASES": ["phase3"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_PHASES": ["phase4"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_PHASES": [], "PREDICTOR_TARGET_TYPES": ["phase"]},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_PHASES": ["phase1"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_PHASES": ["phase2"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_PHASES": ["phase3"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_PHASES": ["phase4"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_PHASES": [], "PREDICTOR_TARGET_TYPES": ["phase"]},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_PHASES": ["phase1"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_PHASES": ["phase2"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_PHASES": ["phase3"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_PHASES": ["phase4"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_PHASES": [], "PREDICTOR_TARGET_TYPES": ["phase"]},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_PHASES": ["phase1"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_PHASES": ["phase2"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_PHASES": ["phase3"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_PHASES": ["phase4"], "PREDICTOR_TARGET_TYPES": ["study_duration", "enrollment_count", "operational_rate"]},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_PHASES": [], "PREDICTOR_TARGET_TYPES": ["phase"]},   
        ]
        
        for task in tasks:
            g.logger.info(f"Sending curl request with {task}")
            result = run_curl_command(task)
            if result.returncode == 0:
                g.logger.info("Curl request sent and received successfully")
            else:
                g.logger.error(f"Curl request failed (code {result.returncode})")
                g.logger.error(f"Detailed error: {result.stderr}")
            time.sleep(10)  # delay to avoid overloading the server
    
    finally:
        # Terminate the WSGI server process
        os.kill(wsgi_process.pid, signal.SIGTERM)
        
    
def run_wsgi_server():
    """ Function to start the WSGI server
    """
    g.logger.info("Starting WSGI server")
    command = ["python", "wsgi.py"]
    return subprocess.Popen(command)


def run_curl_command(task: dict):
    """ Query the clustering API with one set of experimental conditions

    Args:
        task (dict): configuration parameters for this part of the exepriment

    Returns:
        response from the clustering API
    """
    query_dict = {
        "EXPERIMENT_MODE": 2,
        "ENVIRONMENT": "ctgov",
        "LOAD_PARSED_DATA": True,
        "CHOSEN_COND_LVL": 4,
        "CHOSEN_ITRV_LVL": 3,
    }
    query_dict.update(task)
    query_json = json.dumps(query_dict)
    
    command = [
        "curl", "-X", "POST", "http://0.0.0.0:8998/ct-risk/cluster/predict",
        "-H", "Content-Type: application/json",
        "-d", query_json,
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result


if __name__ == "__main__":
    main()
