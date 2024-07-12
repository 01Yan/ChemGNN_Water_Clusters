PFAIMD: Potential-Free Artificial Intelligence Molecular Dynamics
===

Overview


# Contents

* [1 Introduction](#1-introduction)
* [2 Citation](#2-citation)
* [3 Structure of the Repository](#3-structure-of-the-repository)
* [4 Getting Started](#4-getting-started)
  * [4.1 Preparations](#41-preparations)
  * [4.2 Install Packages](#42-install-packages)
  * [4.3 Download Datasets](#43-download-datasets)
  * [4.4 Edit the Configuration File](#44-edit-the-configuration-file)
  * [4.5 Run Training](#45-run-training)
  * [4.6 Apply Your Dataset to this Model](#46-apply-your-dataset-to-this-model)
* [5 Questions](#5-questions)



# 1. Introduction
None

# 2. Citation
None



# 3. Structure of the Repository


```
PFAIMD
┌── PFAIMD/
├────── models/
├────── utils/
├────── scripts/
├── data/
├────── example/
├── dataset/
├────── energy_dataset/
├────── force_dataset/
├── logs
├── saves
├── test
├── config.py
├── LICENSE
├── README.md
├── requirements.txt
└── run.py
```

- `PFAIMD/models/`: folder contains the model scripts
- `PFAIMD/utility/`: folder contains the utility scripts
- `PFAIMD/scripts/`: folder contains the data process scripts
- `data/example`: folder contains the example raw data from 1-H2O to 21-H2O
- `dataset/energy_dataset/`: folder contains the processed data for the energy model
- `dataset/force_dataset/`: folder contains the processed data for the force model
- `logs/`: folder contains the files for logs
-  `saves/`: folder contains saved models and training record figures.
-  `test/`: folder contains test datasets and scripts, as well as MD scripts.
- `config.py`: file of a configuration to be applied
- `LICENSE`: license file
- `README.md`: readme file
- `requirements.txt`: main dependent packages (please follow section 3.1 to install all dependent packages)
- `run.py`: training script



# 4. Getting Started

This project is developed using Python 3.9 and is compatible with macOS, Linux, and Windows operating systems.

## 4.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone 
```

(2) Navigate into the repository.
```shell
~ $ cd ChemGNN
~/ChemGNN $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed virtualenv using you source python script), you can use other virtual environments instead (like conda). This part shows how to set it on your macOS or Linux operating system.
```shell
~/ChemGNN $ python -m venv ./venv/
~/ChemGNN $ source venv/bin/activate
(venv) ~/ChemGNN $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 4.2 Install Packages

(1) Install main dependent packages.
```shell
(venv) ~/ChemGNN $ pip install -r requriements.txt
```

(2) Install packages `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-geometric` manually corresponding to your operating systems and GPU version.

CPU Example:
```shell
(venv) ~/ChemGNN $ pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/ChemGNN $ pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/ChemGNN $ pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/ChemGNN $ pip install torch-geometric
```

CUDA Example (If you are not using CUDA 11.3, please modify the suffix part "cuXXX" of each following url to match your CUDA version):
```shell
(venv) ~/ChemGNN $ pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/ChemGNN $ pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/ChemGNN $ pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/ChemGNN $ pip install torch-geometric
```

## 4.3 Download Datasets

(1) Download our dataset ChemGNN_Dataset (The BibTeX of our dataset is listed in the introduction part). This part shows how to download and unzip it on your macOS or Linux operating system using command line. As an alternative, you can directly input the url `https://github.com/EnzeXu/ChemGNN_Dataset/raw/main/ChemGNN_Dataset.zip` into your browser to start downloading.
```shell
(venv) ~/ChemGNN $ wget https://github.com/EnzeXu/ChemGNN_Dataset/raw/main/ChemGNN_Dataset.zip
```

(2) Extract the datasets and ensure they are placed in the correct destination path.
```shell
(venv) ~/ChemGNN $ unzip -q ChemGNN_Dataset.zip -d ../
```

The Structure of the extracted folder `data/`:

```
ChemGNN
┌── data/
├────── GCN/
├────────── GCN_C1P/
├────────── GCN_C2P/
├────────── GCN_N1C/
├────────── GCN_N1P/
├────────── GCN_N2C/
├────────── GCN_N2P/
├────────── GCN_N3C/
├────────── GCN_N3P/
├────────── GCN_UNDOPED/
├────── HEPTAZINE/
├────── WATER/
├── ...
└── ...
```


## 4.4 Edit the Configuration File

(1) Make a copy from the given example configuration file.
```shell
(venv) ~/ChemGNN $ cp config.py.example config.py
```

(2) Edit your configuration file `config.py` (Please check the file name and place of your configuration file is correct). You can use command line tool `vim` or any other text editor.
```shell
(venv) ~/ChemGNN $ vi config.py
```

Given Example of the configuration file`config.py`:


```python
from ChemGNN import get_config

CONFIGS = {
    'data_config': {
        'main_path': './',  # specify the main path of the project if necessary
        'dataset': 'GCN_C1P',  # 11 available datasets: GCN_C1P, GCN_C2P, GCN_N1C, GCN_N1P, GCN_N2C, GCN_N2P, GCN_N3C, GCN_N3P, GCN_UNDOPED, HEPTAZINE, WATER
        'model': 'ChemGNN',  #  6 available models: ChemGNN, GAT, GCN, GraphSAGE, MPNN, PNA
    },
    'training_config': {
        'device_type': 'cpu',  # 'cpu' or 'gpu'. Note that only when the argument is set to 'gpu' and there are available GPU resources equipped, the training will be executed on the GPU. Otherwise, the training will use CPU.
        'epoch': 400,  # epoch
        'epoch_step': 5,  # minimum epoch period for printing in training
        'batch_size': 128,  # batch_size
        'lr': 0.001,  # learning rate
        'seed': 0,  # random seed
        'train_length_rate': 0.6,  # ratio of the train set with the whole dataset
        'test_length_rate': 0.3,  # ratio of the test set with the whole dataset
    }
}

config = get_config(CONFIGS, "data/const_index.py")  # no need to modify the path here
```

At this step, you have the flexibility to make adjustments to the dataset and model type. You can refer to the comments for a list of available choices.

## 4.5 Run Training

(1) Run Training. Note that if this is the first time you choose a dataset (like `GCN_C1P` in this case), the dataset needs to be processed once. Please input `Y` or `y` to continue.

```shell
(venv) ~/ChemGNN $ python run.py
Processed data not found in ./processed/GCN/GCN_C1P/. Do you want to start processing dataset GCN_C1P? This may take 20-40 minutes. [Y/N]Y
... ...
```

(2) Collect the auto-generated training results in `saves/`.
```shell
(venv) ~/ChemGNN $ ls saves/YYYYMMDD_HHMMSS_f/
all_pred_ceal_no.npy    all_true_ceal_no.npy    loss_last_half.png      loss_last_quarter.png   loss_whole.png
model_last.pt           regression_test.png     regression_train.png    regression_val.png      test_pred.npy
test_true.npy           val_pred.npy            val_true.npy
```

## 4.6 Apply Your Dataset to This Model

Please write you dataset entries into `data/const_index_dic` (both `dataset_list` and `dataset` keywords) and organize your dataset following the architecture of current datasets into `data/`. Then modify the configuration file `config.py` to start training.

# 5. Questions

If you have any questions, please contact xezpku@gmail.com.



