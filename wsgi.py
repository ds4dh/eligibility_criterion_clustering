from app import app


HOST_ADDRESS = "0.0.0.0"
PORT_NUMBER = 8998


def main():
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


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