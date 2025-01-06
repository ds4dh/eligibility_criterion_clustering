import argparse
from app import app


def main():
    # Parse command-line arguments, if any is given
    parser = argparse.ArgumentParser(description="Run the Flask application.")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host address for the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8998, 
        help="Port number for the server (default: 8998)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable Flask debug mode"
    )
    args = parser.parse_args()
    
    # Run the Flask application
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


# gunicorn --bind 0.0.0.0:8998 --threads=12 --certfile=utils/cert.pem --keyfile=utils/key.pem wsgi:app


# *** EXAMPLE JSON REQUEST ***
# {
#     "ENVIRONMENT": "ctxai_dev",
#     "DATA_PATH": "Test/1/metadata/intervention/intervention_similar_trials.xlsx",
#     "USER_ID": "gh30298h6g356",
#     "PROJECT_ID": "f784h30f7j9if",
#     "USER_FILTERING": "intervention",
#     "EMBEDDING_MODEL_ID": "pubmed-bert-sentence",
#     "CLUSTER_DIM_RED_ALGO": "umap",
#     "CLUSTER_RED_DIM": 10,
#     "PLOT_DIM_RED_ALGO": "tsne",
#     "PLOT_RED_DIM": 2,
#     "CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY": "utils/api-key.txt",
#     "CLUSTER_REPRESENTATION_MODEL": null,
#     "CLUSTER_REPRESENTATION_GPT_PROMPT": "I have a topic that contains the following documents: \n[DOCUMENTS]\nThe topic is described by the following keywords: \n[KEYWORDS]\nBased on the information above, extract a short but highly descriptive topic label of at most 5 words.\nMake sure it is in the following format: <topic type>: <topic label>, where <topic type> is either 'Inclusion criterion: ' or 'Exclustion criterion: '\n"
# }