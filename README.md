# MisEdit
**MisEdit: Evidence that Rewrite Metrics May Overestimate Misconception Correction for ROME-Based Editing in Sub-3B Language Models**

## Overview

MisEdit investigates whether successful knowledge edits measured by rewrite accuracy correspond to genuine changes in model beliefs. Using MythBench, a benchmark of 48 validated parametric misconceptions, we evaluate ROME-based editing on TinyLlama-1.1B and Qwen2.5-1.5B.

Our findings show that rewrite accuracy frequently improves after editing, while multiple-choice belief probing and free-form generation remain unchanged, revealing a mismatch between rewrite-based evaluation and belief-oriented evaluation.

## Repository Structure

### Code

* `MisEdit_Kaggle_Full.ipynb`

  * Full TinyLlama-1.1B evaluation on all 48 MythBench misconceptions.

* `MisEdit_LightningAI.ipynb`

  * Qwen2.5-1.5B stratified replication experiment.

* `MisEdit_Extensions.ipynb`

  * Additional analyses including:

    * MC v1 vs MC v2 comparison
    * Token probability analysis
    * Statistical power analysis
    * Auxiliary factual edit study
    * Exploratory MEMIT experiments

### Results

Contains the CSV files used to generate the tables and figures reported in the paper.

### Dataset

MythBench v10 benchmark used for misconception evaluation.

## Models

* TinyLlama/TinyLlama-1.1B-Chat-v1.0
* Qwen/Qwen2.5-1.5B-Instruct

## Editing Method

* ROME (EasyEdit Framework)

## Hardware

* Kaggle T4 GPU (TinyLlama experiments)
* Lightning.ai L4 GPU (Qwen experiments)

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in the following order:

1. MisEdit_Kaggle_Full.ipynb
2. MisEdit_LightningAI.ipynb
3. MisEdit_Extensions.ipynb

## Citation

Samuel Stephen and R. Vignesh.

*MisEdit: Evidence that Rewrite Metrics May Overestimate Misconception Correction for ROME-Based Editing in Sub-3B Language Models* (2026).
