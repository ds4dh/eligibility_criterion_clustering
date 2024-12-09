# Start with a lightweight CUDA image with support for Python
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Set root privileges for installs
USER root

# Install system build dependencies and Miniconda
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

# Add Miniconda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment and install cuML, Python, and CUDA libraries
RUN conda create --solver=libmamba -n ctxai \
      -c rapidsai -c conda-forge -c nvidia \
      cuml=23.12 python=3.10 cuda-version=11.8 -y \
    && conda clean -afy

# Activate the environment and install pip requirements
COPY environment/pip_requirements.txt /app/environment/pip_requirements.txt
RUN /bin/bash -c "source activate ctxai \
    && pip install --no-cache-dir -U -r /app/environment/pip_requirements.txt \
    && pip install --no-cache-dir gunicorn \
    && conda env export > /app/environment/environment_droplet.yml"

# Set the environment for the runtime user
ENV PATH="/opt/conda/envs/ctxai/bin:$PATH"
ENV CONDA_DEFAULT_ENV=ctxai

# Default command
CMD ["/bin/bash"]