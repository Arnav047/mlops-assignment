# Pedestrian Detection Pipeline: Reproducibility Guide

This guide outlines how any engineer can identically replicate the training and evaluation experiments for this pedestrian object detection pipeline.

## 1. Environment Setup
To ensure identical software dependency versions and avoid package conflicts, initialize a virtual environment and install the pinned dependencies from the root directory:

```bash
# Create and activate a clean virtual environment
python -m venv .venv
.venv\Scripts\activate 

# Install identical package versions
pip install -r requirements.txt


Step A: Download the Dataset
python download_data.py


Step B: Run Baseline Training (Experiment 1)
python main.py

Step C: Run Alternative Training (Experiment 2)
Change a setting inside base_config.yaml (like modifying the learning rate lr), save the file, and run the pipeline again:
python main.py


3. View the MLflow Dashboard
To launch the visual dashboard and compare the training runs side-by-side, run:
mlflow ui --backend-store-uri sqlite:///mlflow.db

4. Run via Docker
To build and run the entire pipeline inside an isolated container, use these two commands:

Build the Image:
docker build -t mlops-assignment .

Run the Pipeline:
docker run mlops-assignment

5. Automated CI/CD Workflow (GitHub Actions)
This project includes a continuous integration safety net configured in .github/workflows/ci.yml.

Every time we run git push to upload code to the main branch on GitHub, an automated cloud runner instantly triggers to:

Provision a fresh, isolated Ubuntu Linux environment.

Clone the repository files.

Build the Dockerfile from scratch to verify absolute code reproducibility and catch environment-breaking bugs automatically.
