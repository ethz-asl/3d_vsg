# 3D Variable Scene Graphs
This repository will contain the code for **DeltaVSG**, our framework to estimate **3D Variable Scene Graphs (VSG)** for long-term semantic scene change prediction.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/36043993/190389702-9f838aa8-2876-4086-9f0a-431a51bccc1d.png">
</p>
Scene variability prediction using DeltaVSG.

# Table of Contents
**Credits**
* [Paper](#Paper)

**Setup**
* [Installation](#Installation)
* [Data](#Data)

**Examples**
<!-- no toc -->
* [Training the Model](#training-the-model)
* [3DVSG Inference](#3dvsg-inference)
* [Evaluating an Experiment](#evaluating-an-experiment)


# Paper
If you find this useful for your research, please consider citing our paper:

* Samuel Looper, Javier Rodriguez-Puigvert, Roland Siegwart, Cesar Cadena, and Lukas Schmid, "**3D VSG: Long-term Semantic Scene Change
Prediction through 3D Variable Scene Graphs**", accepted for *IEEE International Conference on Robotics and Automation (ICRA)*, 2023. \[ [IEEE](https://ieeexplore.ieee.org/document/10161212) | [ArXiv](https://arxiv.org/abs/2209.07896) \]
  ```bibtex
  @inproceedings{looper22vsg,
  author = {Looper, Samuel and Rodriguez-Puigvert, Javier and Siegwart, Roland and Cadena, Cesar and Schmid, Lukas},  
  title = {3D VSG: Long-term Semantic Scene Change Prediction through 3D Variable Scene Graphs},
  publisher = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2023},
  doi = {10.1109/ICRA48891.2023.10161212},
  }
  ```
  
# Setup 
## Installation
1. Clone the repository using [SSH Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh):
    ```bash
    export REPO_PATH=<path/to/destination>
    cd $REPO_PATH
    git clone git@github.com:ethz-asl/3d_vsg.git
    cd 3d_vsg
    ```
2. Create a Python environment. We recommend using [conda](https://docs.conda.io/en/latest/):
    ```bash
    conda create --name 3dvsg python=3.8
    conda activate 3dvsg
    pip install -r requirements.txt
    
    ```
    > __Note__ The installation is configured for CPU-version of torch. If you have cuda replace `cpu` in the above instructions and in `requirements.txt` with your cuda version, e.g. `cu102` for CUDA 10.2.

3. You're all set!

## Data Setup

### Downloading the Data
The dataset used in our experiments is based on the [3RScan Dataset](https://waldjohannau.github.io/RIO/) [1] and [3D SSG Dataset](https://3dssg.github.io/) [2].

> [1] Wald, Johanna, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner, "Rio: 3d object instance re-localization in changing indoor environments", in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 7658-7667. 2019.
>
> [2] Wald, Johanna, Helisa Dhamo, Nassir Navab, and Federico Tombari, "Learning 3d semantic scene graphs from 3d indoor reconstructions", in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 3961-3970. 2020.

**Option 1**: Download the pre-processed 3D VSG Embeddings.
  
The pre-processed training and evaluation data for the examples is available for donwload from [GDrive](https://drive.google.com/drive/folders/1Fmgvvj26SPJCQ51piuwIxj4Za2rS-iPT?usp=share_link):

    ```bash
    cd $REPO_PATH/3d_vsg
    mkdir data
    cd data
    gdown 1ub_pdt7vXIlJVK0V4B_ydWSKiuX1wTP_
    unzip processed.zip
    rm processed.zip
    ```
**Option 2**: Download the original data and process it.

1. Sign up for the [3R Scan Dataset](https://waldjohannau.github.io/RIO/download) to get access to the data and  `download_3rscan.py` script.

2. Download the semantic segmentation data:

    ```bash
    cd $REPO_PATH/3d_vsg
    mkdir -p data/raw
    python download_3rscan.py --out_dir data/raw/semantic_segmentation_data --type 'semseg.v2.json'
    ```

3. Download the meta data file `3RScan.json` and place it in `data/raw`.
  
4. Download the 3DSSG annotations:
  
    ```bash
    wget http://campar.in.tum.de/public_datasets/3RScan/3RScan.json -P data/raw
    wget https://campar.in.tum.de/public_datasets/3DSSG/3DSSG.zip 
    unzip 3DSSG.zip -d data/raw
    rm 3DSSG.zip
    ```

5. Process the raw data to get the 3D VSG Embeddings by setting `load` in `config/DatasetCfg.py` to `false` and running:

    ```bash
    python -m src.scripts.generate_dataset
    ```

> __Note__ Any pre-processed dataset files currently in `data/processed` will be moved to `data/old_processed` and timestamped. The newly created dataset will generate files in `dataset/processed`.

### Downloading the Pretrained Models
The pre-trained network weights are available for download on [GDrive](https://drive.google.com/drive/folders/1Fmgvvj26SPJCQ51piuwIxj4Za2rS-iPT?usp=share_link):

```bash
cd $REPO_PATH/3d_vsg
mkdir pretrained
gdown 1hHmXSXtAUqqGNMn4vsEvc3XA3SLxpV4o
unzip models.zip -d pretrained
rm models.zip
```

# Examples

## Training the Model
Before starting training, make sure you have setup the data as explained [above](#downloading-the-data). To train a new model, run:

```
python -m src.scripts.train_variability
```

> __Note__ Additional dataset parameters can be configured in `config/DatasetCfg.py`. Addtional model parameters can be configured in the hyperparameter dictionary in `src/scripts/train_variability.py`.


## 3DVSG Inference
To infere 3D Variable Scene Graphs, if not already done so setup the output directory and download the data splits:
```bash
cd $REPO_PATH/3d_vsg
mkdir results
cd results
gdown 1mT-agKOkB8ebg6NsliOnIReRO81PjHQL 
gdown 1jO4rG1qlYj7MHqxNx-6Ql_igv3lsi_79
```

Then, to run model inference run:
```bash
python -m src.scripts.inference
```

> __Note__ Additional dataset parameters can be configured in `config/DatasetCfg.py` in the `InferenceCfg` subclass. Addtional model parameters can be configured in the hyperparameter dictionary in `src/scripts/inference.py`.


## Evaluating an Experiment
To evaluate the performance of a 3DVSG model run:

```bash
python -m src.scripts.eval
```

> __Note__ The splits path, dataset root, model weights path, and hyperparameter dictionary can be configured in `src/scripts/eval.py`.
