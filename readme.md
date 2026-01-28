# Read and Visualize 3D Scene Graphs Tutorial

The objective of this repository is to teach you how to read visualize and save in JSON format the 3D Scene Graphs from 3DSSG dataset.

### Prerequisites
- Install [Minicoda](https://www.anaconda.com/docs/getting-started/miniconda/main) and create a conda env

```
conda create -n 3dssg python=3.10
conda activate 3dssg
```
- Install the requirements
```
pip install -r requirements.txt
```

### Download the dataset

- 3DSSG is already inside ./3DSSG
- Download 3RScan with

```
python download_3RScan.py -o 3RScan
```

### Notebooks

- study the notebook familiarize_3dssg.ipynb