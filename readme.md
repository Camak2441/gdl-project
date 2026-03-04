# Scene Graph Query Answering with GNNs

This repository is the code for Scene Graph Query Answering with GNNs. 

### Prerequisites
- Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) and create a conda env

```bash
conda create -n 3dssg python=3.10
conda activate 3dssg
```
- Install the requirements
```bash
pip install -r requirements.txt
```

### Download the dataset

- 3DSSG is already inside ./3DSSG
- Download 3RScan with

```bash
python download_3RScan.py -o 3RScan
```

### Scripts

Before running any scripts, run the following:
```bash
cd scripts # Set the current directory to the scripts directory
. ./setup.sh # Sets up any needed environment variables
```

To train a model, run 
```bash
python train_model.py [model_name] [dataset]
```

The set of model and dataset shorthands can be found in `/src/models/__init__.py` and `/src/data/__init__.py`.

To programmatically generate questions, run
```bash
python generate_programmatic_qs.py
```

To generate LLM written questions, run
```bash
python generate_complex_qs.py
```

To embed a dataset, run
```bash
python questions_to_dataset.py
```

To choose your dataset location, you will need to alter the paths in the script itself. 

### Notebooks

This project also features the notebook `notebooks/test_3dssg_model.ipynb`, which allows you to query the LLM, and visualise the answers. It also requires the packages installed into the conda environment. 