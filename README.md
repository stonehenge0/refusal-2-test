# BabelSteering

**Submission ID:** 1570
**Link:** https://openreview.net/forum?id=SMk5KJ6l8C#discussion
**Date:** 18.09.2026
**Venue:** EACL2027 Athens

This repository contains the code and data used for the experiments in our paper *"BabelSteering: Multilingual Safety Alignment via English Steering Vectors"*.

To reproduce our experiments, run the setup script and reproduce our results as described below.

## Setup
Run the setup script

```bash
git clone TODO
cd refusal_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models). It will then set up a virtual environment and install the required packages.

Then, install the evaluation harness from source. 

```bash
cd lm-evaluation-harness
pip install -e .
```
## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python -m pipeline.run_pipeline --config_path "$CONFIG"
```

where `$CONFIG` is a path to a configuration file, like the ones provided in the appendix of the paper. Feel free to modify the HuggingFace paths and hyperparameter settings to run the desired experiments.

## Required resources

We recommend running this on an HPC cluster and provide an example SLURM script in `run_pipe.bash`. Experiments require an A100 80GB or similar hardware.

## Anonymity
Some parts of the repo (file paths, cluster names, comments with names in them) were redacted for anonymity.