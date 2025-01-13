#!/bin/bash

# Environment specifics
ENV_NAME="ctxai"
PYTHON_VERSION="3.10"

# Create a new conda environment with cuML and activate it
conda create --solver=libmamba -n $ENV_NAME \
      -c rapidsai -c conda-forge -c nvidia \
      cuml=23.12 python=$PYTHON_VERSION cuda-version=11.8 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Check if the activation was successful
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then

    # Install latest version of pip packages
    pip install -U -r environment/pip_requirements.txt
    pip install gunicorn
    conda env export > environment/environment_droplet.yml

    # Success message
    echo -e "\nSuccessfully installed all packages in $ENV_NAME.\n\
    Don't forget to activate it by using the following command:\n\
    >>> conda activate $ENV_NAME\n"

# Avoid installing anything if still in the base environment
else
    echo "Failed to activate the environment. No packages were installed."
fi
