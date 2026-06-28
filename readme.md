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