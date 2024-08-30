# CTxAI - Eligibility Criteria

This is the repository for the manuscript "Analysis of Eligibility Criterion Clusters Based on Large Language Models for Clinical Trial Design".

![Pipeline](images/pipeline.png)

## Getting Started

### Install Dependencies

#### Using Docker

1. Ensure Docker and Docker Compose are installed on your system. You can download them from [Docker&#39;s official website](https://www.docker.com/get-started).
2. Clone the repository to your local machine:

   ```
   git clone https://github.com/albornet/ctxai.git  # or git@github.com:albornet/ctxai.git
   cd ctxai
   ```
3. To build and start the project using Docker Compose, run:

   ```
   docker-compose up --build  # or docker-compose build, followed by docker-compose up
   ```

   This will build the Docker images and starts the service defined in `docker-compose.yml`. It's a simple way to get the environment set up without manually installing dependencies, and to interact with the pipeline as a webservice (see below).

#### Installing Environment with Droplet

If you prefer not to use Docker or if you're working in an environment where Docker is not available, you can install the dependencies manually using Conda:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda on your system if you haven't already.
2. Create a new Conda environment using the `environment/setup_env.sh` script:

   ```
   ./environment/setup_env.sh
   ```
3. Activate the new environment:

   ```
   conda activate ctxai
   ```

### Using the repository

#### Running the experiments

Start by downloading the raw dataset by executing:

```
./download_raw_data.sh
```

Then, build the eligibility criterion dataset by executing:

```
python src/parse_data.py
```

Then, run all experiments:

```
python experiments/experiment_1.py
python experiments/experiment_2.py
./experiments/experiment_3.sh
```

Finally, plot the results:

```
python experiments/plot_experiment_1.py
python experiments/plot_experiment_2.py
python experiments/plot_experiment_3.py
```

#### As a webservice (using Docker)

To execute the pipeline using Docker, send a POST request with the desired configuration. Note that "RAW_DATA_PATH", "USER_ID", and "PROJECT_ID" are mandatory fields in the request. Also note that "RAW_DATA_PATH" can also be a file or a directory. In the latter case, all ".xlsx" files in the directory will be read. You can add any field that you want to update in config.yaml:

```json
{
    "RAW_DATA_PATH": "data/ctxai/eligibility_criteria.xlsx",
    "USER_ID": "1234",
    "PROJECT_ID": "5678",
    "EMBEDDING_MODEL_ID": "pubmed-bert-sentence",
    "CLUSTER_DIM_RED_ALGO": "tsne",
    "CLUSTER_RED_DIM": 2,
    "PLOT_DIM_RED_ALGO": "tsne",
    "PLOT_RED_DIM": 2,
    "CLUSTER_REPRESENTATION_MODEL": "gpt",
    "CLUSTER_REPRESENTATION_GPT_PROMPT": "I have a topic that contains the following documents:\n[DOCUMENTS]\nThe topic is described by the following keywords:\n[KEYWORDS]\nBased on the information above, extract a short but highly descriptive topic label of at most 5 words.\nMake sure it is in the following format: <topic type>: <topic label>, where <topic type> is either 'Inclusion criterion: ' or 'Exclustion criterion: '"
}
```

You can use ARC to send your request to http://localhost:8984/ct-risk/cluster/predict, or you can write your configuration to a file (e.g., `request.json`), and execute:

```bash
curl -X POST -H "Content-Type: application/json" -d @request.json http://localhost:8984/ct-risk/cluster/predict
```

After you send the request, the container will execute the pipeline and respond with an output like this one:

```json
{
    "cluster_json_path": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/ec_clustering.json",
    "cluster_raw_ec_list_path": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/raw_ec_list.csv",
    "cluster_visualization_paths": {
        "all": {
            "html": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_all.html",
            "png": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_all.png"
        },
        "top_20": {
            "html": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_top_20.html",
            "png": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_top_20.png"
        }
    }
}
```

The response defines where the results (included formatted cluster output, as well as interactive visualizations) were saved.

#### Direct Execution

If you've set up your environment using Conda:

1. Ensure all configurations are correctly set in `config.yaml` according to your project's needs.
2. From the project's root directory, run:

   ```
   python src/parse_data.py
   python src/cluster_data.py
   ```
