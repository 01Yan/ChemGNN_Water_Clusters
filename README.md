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
~ $ cd PFAIMD
~/PFAIMD $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed virtualenv using you source python script), you can use other virtual environments instead (like conda). This part shows how to set it on your macOS or Linux operating system.
```shell
~/PFAIMD $ python -m venv ./venv/
~/PFAIMD $ source venv/bin/activate
(venv) ~/PFAIMD $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 4.2 Install Packages

(1) Install main dependent packages.
```shell
(venv) ~/PFAIMD $ pip install -r requriements.txt
```

(2) Install packages `torch-scatter`, `torch-sparse`, `torch-cluster` and `torch-geometric` manually corresponding to your operating systems and GPU version.

CPU Example:
```shell
(venv) ~/PFAIMD $ pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/PFAIMD $ pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/PFAIMD $ pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
(venv) ~/PFAIMD $ pip install torch-geometric
```

CUDA Example (If you are not using CUDA 11.3, please modify the suffix part "cuXXX" of each following url to match your CUDA version):
```shell
(venv) ~/PFAIMD $ pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/PFAIMD $ pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/PFAIMD $ pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
(venv) ~/PFAIMD $ pip install torch-geometric
```

## 4.3 Download Datasets



## 4.4 Edit the Configuration File

(1) Edit your configuration file `config.py` (Please check the file name and place of your configuration file is correct). You can use command line tool `vim` or any other text editor.
```shell
(venv) ~/PFAIMD $ vi config.py
```

Given Example of the configuration file`config.py`:


```python
from PFAIMD import get_config

CONFIGS = {
    'data_config': {
        'main_path': './',
        'dataset': 'energy_dataset',
        'model': 'ChemGNN_energy',
    },
    'training_config': {
        'device_type': 'cpu',
        'loss_fn_id': 1,
        'epoch': 20,
        'epoch_step': 1,
        'batch_size': 1024,
        'lr': 0.001,
        'seed': 0,
        'train_length_rate': 0.7,
        'val_length_rate': 0.2,
    }
}

DATA_PROCESSED_CONFIGS = {
    'generate_d_bt_config': {
        'base_dir': 'example',
        'pattern': '*WATER',
        'H_H_cutoff': 1.6,
        'H_O_cutoff': 2.4,
        'O_O_cutoff': 2.8
    },
    'processed_dataset_config':{
        'processed_src_path': 'data/example',
        'processed_dst_path': 'dataset'
    }
}

config = get_config(CONFIGS, DATA_PROCESSED_CONFIGS)
```

At this step, you have the flexibility to make adjustments to the dataset and model type. You need to choose whether to train the energy model or the force model. 

- If `'dataset': 'energy_dataset'` and `'model': 'ChemGNN_energy'`, it indicates that the energy dataset is loaded and the energy model is trained.
- If `'dataset': 'force_dataset'` and `'model': 'ChemGNN_force'`, it indicates that the force dataset is loaded and the force model is trained.


## 4.5 Run Training

(1) Run Training. Note that if you have downloaded our data sets and saved them in the corresponding path, you do not need to process them anymore.

```shell
(venv) ~/PFAIMD $ python run.py
```

(2) Collect the auto-generated training results in `saves/`.
```shell
(venv) ~/PFAIMD $ ls saves/YYYYMMDD_HHMMSS_f/
loss_last_half.png      loss_last_quarter.png    loss_whole.png        model_last.pt           test_pred.npy
test_true.npy
```

## 4.6 Run Testing

(1) Run Testing. Note that we saved the trained energy and force models in test/energy.pt and test/forces.pt. Feel free to testing them.
```shell
(venv) ~/PFAIMD $ cd test
(venv) ~/PFAIMD $ python mdchemgnn.py
```

If you want to test your retrained model, please rename the energy model to `'energy.pt'` and the force model to `'forces.pt'`, and replace the existing two model files.

## 4.7 Apply Your Dataset to This Model
None

# 5. Questions

If you have any questions, please contact .



