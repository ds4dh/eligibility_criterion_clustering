import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template, send_file, redirect
from typing import Any
from src import (
    config,
    parse_data_fn,
    cluster_data_fn,
    run_experiment_1,
    run_experiment_2,
    create_visualizations_from_ct_paths_or_nct_ids,
    create_visualization_from_ct_info,
)
config.load_default_config()
logger = config.CTxAILogger("INFO")

app = Flask(__name__, template_folder="templates", static_folder="static")

PREFIX_PATH = "/ct-risk/cluster"
PREDICT_PATH = "%s/predict" % PREFIX_PATH
VISUALIZE_PATH = "%s/visualize" % PREFIX_PATH
SERVE_HTML_PATH = "%s/serve-html" % PREFIX_PATH


@app.route("/", methods=["GET"])
def index():
    """ Serve the HTML form for user input
    """
    return render_template("index.html")


@app.route(rule=SERVE_HTML_PATH, methods=["GET"])
def serve_html():
    """ Serve the visualization HTML file
    """
    html_path = request.args.get("path")
    if html_path and os.path.exists(html_path):
        logger.info(f"Serving HTML file at {html_path}")
        return send_file(html_path, mimetype="text/html")
    else:
        logger.error(f"Requested HTML file not found at {html_path}")
        return jsonify({"error": "File not found"}), 404


@app.before_request
def redirect_to_https():
    """ Redirect HTTP requests to HTTPS
    """
    if not request.is_secure:
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)
    

@app.route("/get-latest-log", methods=["GET"])
def get_latest_log():
    """ Serve the latest log line
    """
    log_file_path = "logs/app.log"
    if not os.path.exists(log_file_path):
        return jsonify({"log": "Log file not found"}), 404
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            last_line = lines[-1] if lines else "Log file is empty"
            return jsonify({"log": last_line}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route(rule=VISUALIZE_PATH, methods=["POST"])
def visualize():
    """ Handle visualization logic for a given request and demonstration mode
    """
    # Validate request data
    request_data = request.get_json(force=True)
    demo_response, demo_value = check_request_for_demo(request_data)
    if demo_response is not None:
        return demo_response, demo_value
    logger.info(f"Starting visualization pipeline for mode {demo_value}")
    
    # Generate visualizations based on demo mode value
    visu_response, visu_value = generate_visualization(demo_value, request_data)
    if visu_response is not None:
        return visu_response, visu_value
    logger.info(f"Visualization successfully generated for mode {demo_value}")
    
    # Return the path to the visualization HTML file
    html_path = visu_value["html"]
    if os.path.exists(html_path):
        logger.info(f"Generated visualization at {html_path}")
        return jsonify({"html_path": f"{SERVE_HTML_PATH}?path={html_path}"})
    else:
        logger.error(f"Visualization file not found at {html_path}")
        return jsonify({"error": "Visualization file not found"}), 404


def check_request_for_demo(request_data: dict[str, Any]):
    """ Check that required fields are in the request data, given situation
    """
    # Extract demo mode
    if "DEMO_MODE" not in request_data:
        return jsonify({"error": "Missing DEMO_MODE in request data"}), 400
    demo_mode = request_data["DEMO_MODE"]
    
    # Check fields for visualization from a clinical trial file
    if demo_mode == "ct_file":
        if "TARGET_CT_DICT" not in request_data:
            return jsonify({"error": "Missing TARGET_CT_DICT in request data"}), 400
    
    # Check fields for visualization from the NCT-ID of a clinical trial
    elif demo_mode == "nct_id":
        if "TARGET_NCT_ID" not in request_data:
            return jsonify({"error": "Missing TARGET_NCT_ID in request data"}), 400
    
    # Check fields for visualization from lists of phase(s), cond(s), itrv(s)
    elif demo_mode == "ct_info":
        required_keys = ["CHOSEN_PHASES", "CHOSEN_COND_IDS", "CHOSEN_ITRV_IDS"]
        if not all([k in request_data for k in required_keys]):
            return jsonify({"error": "Missing field in request data"}), 400
    
    # Still able to provide error information to the client
    else:
        return jsonify({"error": "Invalid DEMO_MODE"}), 400
    
    # In case all is good, no response is emmited, and value is demo mode
    return None, demo_mode


def generate_visualization(demo_value: str, request_data: dict[str, Any]):
    """ Generate visualization for a given demonstration mode using request data
    """
    try:
        # Visualization from a clinical trial protocol file
        if demo_value == "ct_file":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
                temp_file.write(json.dumps(request_data["TARGET_CT_DICT"]).encode('utf-8'))
                ct_path = temp_file.name
                
            visu_output = create_visualizations_from_ct_paths_or_nct_ids(
                ct_file_paths=[ct_path],
                n_examples_to_generate=1,
            )[0]
        
        # Visualization from the NCT-ID of a clinical trial 
        elif demo_value == "nct_id":
            visu_output = create_visualizations_from_ct_paths_or_nct_ids(
                ct_nct_ids=[request_data["TARGET_NCT_ID"]],
                n_examples_to_generate=1,
            )[0]
        
        # Visualization from a set of phase(s), condition(s), and intervention(s)
        elif demo_value == "ct_info":
            visu_output = create_visualization_from_ct_info(
                ct_phases=request_data["CHOSEN_PHASES"],
                cond_ids=request_data["CHOSEN_COND_IDS"],
                itrv_ids=request_data["CHOSEN_ITRV_IDS"],
                ct_path="custom_ct_info",
            )
        
        # Return no error response and the required visualization output
        visu_response = None
        return visu_response, visu_output
    
    # Return error message if visualiation failed for some reason
    except RuntimeError as visualisation_error:
        return jsonify({"error": str(visualisation_error)}), 400


@app.route(rule=PREDICT_PATH, methods=["POST"])
def predict():
    """ Cluster data found in data_dir found in request's field
    """
    # Initialization and validation of required fields in JSON payload
    logger.info("Starting eligibility criteria clustering pipeline")
    request_data = request.get_json(force=True)
    if "EXPERIMENT_MODE" in request_data:
        logger.info("Experiment %1i being run" % request_data["EXPERIMENT_MODE"])
    else:
        required_keys = [
            "ENVIRONMENT",
            "DATA_PATH",
            "USER_ID",
            "PROJECT_ID",
            "USER_FILTERING",
            "EMBEDDING_MODEL_ID",
        ]
        if not all([k in request_data for k in required_keys]):
            return jsonify({"error": "Missing field in request data"}), 400
    
    # Update in-memory configuration using request data
    config.update_config(request_data)
    
    # Parse raw data into pre-processed data files
    logger.info("Parsing criterion texts into individual criteria")
    parse_data_fn()
    
    # Perform one of the experiments
    if "EXPERIMENT_MODE" in request_data:
        exp_id = request_data["EXPERIMENT_MODE"]
        if exp_id == 1:
            run_experiment_1()
        elif exp_id == 2:
            run_experiment_2()
        return jsonify({"status": "success"}), 200
    
    # Or simply cluster requested data (ctgov or ctxai)
    else:
        logger.info("Clustering procedure started")
        cluster_output = cluster_data_fn(request_data["EMBEDDING_MODEL_ID"])
    
    # Return jsonified file paths corresponding to the written data and plot
    logger.info("Success!")
    return jsonify({
        "cluster_json_path": cluster_output.json_path,
        "cluster_visualization_paths": cluster_output.visualization_paths,
        "cluster_raw_ec_list_path": cluster_output.raw_ec_list_path,
    }), 200


if __name__ == "__main__":
    host_address = "0.0.0.0"
    port_number = 8984
    app.run(debug=False, host=host_address, port=port_number)