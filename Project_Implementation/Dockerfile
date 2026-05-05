# PyTorch 2.7.0 + CUDA 12.8 — required for NVIDIA Blackwell (RTX 5050, sm_120)
# PyTorch 2.1.0 only supports up to sm_90; Blackwell requires 2.7+
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Set working directory (matches volume mount in docker-compose.yml)
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install strictly audited dependencies from Roadmap
# Install stable, compatible dependencies (pinned)

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1 \
    scikit-learn==1.5.2 \
    imbalanced-learn==0.12.3 \
    optuna==3.6.1 \
    plotly \
    tqdm \
    requests \
    mapie==0.8.6 \
    shap==0.44.1 \
    fastapi \
    "uvicorn[standard]" \
    chembl-webresource-client \
    jupyterlab \
    ipywidgets

# RDKit from pip works perfectly on Linux containers
RUN pip install --no-cache-dir rdkit==2024.03.1

# Install SYBA directly from GitHub source
RUN pip install --no-cache-dir git+https://github.com/lich-uct/syba.git

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab on run
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=tox21"]