import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
import time
import json
import signal
import subprocess
from src import config
config.load_default_config()
logger = config.CTxAILogger("INFO")


def run_wsgi_server():
    """ Function to start the WSGI server
    """
    logger.info("Starting WSGI server")
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
        "EXPERIMENT_MODE": 1,
        "ENVIRONMENT": "ctgov",
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


def main():
    """ Run the clustering pipeline for many different experimental conditions
    """
    # Start the WSGI server
    wsgi_process = run_wsgi_server()
    
    # Give the server some time to start
    logger.info(f"Experiment 1 will start in a few seconds")
    time.sleep(20)
    
    try:
        # Define your sequence of environments and data paths
        tasks = [
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_COND_LVL": 2, "CHOSEN_ITRV_LVL": 1},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_COND_LVL": 3, "CHOSEN_ITRV_LVL": 2},
            {"CHOSEN_COND_IDS": ["C01"], "CHOSEN_COND_LVL": 4, "CHOSEN_ITRV_LVL": 3},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_COND_LVL": 2, "CHOSEN_ITRV_LVL": 1},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_COND_LVL": 3, "CHOSEN_ITRV_LVL": 2},
            {"CHOSEN_COND_IDS": ["C04"], "CHOSEN_COND_LVL": 4, "CHOSEN_ITRV_LVL": 3},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_COND_LVL": 2, "CHOSEN_ITRV_LVL": 1},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_COND_LVL": 3, "CHOSEN_ITRV_LVL": 2},
            {"CHOSEN_COND_IDS": ["C14"], "CHOSEN_COND_LVL": 4, "CHOSEN_ITRV_LVL": 3},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_COND_LVL": 2, "CHOSEN_ITRV_LVL": 1},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_COND_LVL": 3, "CHOSEN_ITRV_LVL": 2},
            {"CHOSEN_COND_IDS": ["C20"], "CHOSEN_COND_LVL": 4, "CHOSEN_ITRV_LVL": 3},
        ]
        
        for i, task in enumerate(tasks):
            task.update({"LOAD_PARSED_DATA": i != 0})
            # Use the line below after completing experiment 1 with the line above
            # if you want to generate cluster representations with GPT-3.5-Turbo 
            # task.update({
            #     "LOAD_PARSED_DATA": True,
            #     "LOAD_EMBEDDINGS": True,
            #     "LOAD_BERTOPIC_RESULTS": True,
            #     "CLUSTER_REPRESENTATION_MODEL": "gpt",
            #     "REGENERATE_REPRESENTATIONS_AFTER_LOADING_BERTOPIC_RESULTS": True,
            # })
            logger.info(f"Sending curl request with {task}")
            result = run_curl_command(task)
            if result.returncode == 0:
                logger.info("Curl request sent and received successfully")
            else:
                logger.error(f"Curl request failed (code {result.returncode})")
                logger.error(f"Detailed error: {result.stderr}")
            time.sleep(10)  # delay to avoid overloading the server
    
    finally:
        # Terminate the WSGI server process
        os.kill(wsgi_process.pid, signal.SIGTERM)
        

if __name__ == "__main__":
    main()
