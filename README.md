ChemGNN: Chemical Environment Graph Neural Network
===

<p>
  <img src="https://github.com/chenm19/ChemGNN/assets/90367338/0ebecbac-f0bb-4347-8a6c-b89e3a42277b" alt="ChemGNN_Dataset_Logo" width="100%">
</p>

Overview of Chemical Environment Graph Neural Network (ChemGNN) for optical band gap prediction of g-C3N4 nanosheet. Chemical Environment Adaptive Learning (CEAL) layers are employed to extract messages from atoms' chemical environments.


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
This study presents a novel Machine Learning Algorithm, named Chemical Environment Graph Neural Network (ChemGNN), designed to accelerate materials property prediction and advance new materials discovery. Graphitic carbon nitride (g-C3N4) and its doped variants have gained significant interest for their potential as optical materials. Accurate prediction of their band gaps is crucial for practical applications, however, traditional quantum simulation methods are computationally expensive and challenging to explore the vast space of possible doped molecular structures. The proposed ChemGNN leverages the learning ability of current graph neural networks (GNNs) to satisfactorily capture the characteristics of atoms' local chemical environment underlying complex molecular structures. Our benchmark results demonstrate more than 100% improvement in band gap prediction accuracy over existing GNNs on g-C3N4. Furthermore, the general ChemGNN model can precisely foresee band gaps of various doped g-C3N4 structures, making it a valuable tool for performing high-throughput prediction in materials design and development. 

# 2. Citation

If you use our code or datasets from `https://github.com/EnzeXu/ChemGNN_Dataset` for academic research, please cite the following paper and the following dataset:

Paper BibTeX:

```
@article{chen2023chemical,
  title        = {Chemical Environment Adaptive Learning for Optical Band Gap Prediction of Doped Graphitic Carbon Nitride Nanosheets},
  author       = {Chen, Chen and Xu, Enze and Yang, Defu and Yin, Haibing and Wei, Tao and Chen, Hanning and Wei, Yong and Chen, Minghan},
  journal      = {arXiv preprint arXiv:2302.09539},
  year         = {2023}
}
```

Dataset BibTeX:

```
@misc{chen2023chemgnndatasets,
  title        = {Chemical Environment Graph Neural Network (ChemGNN) Datasets},
  author       = {Chen, Chen and Xu, Enze and Yang, Defu and Yin, Haibing and Wei, Tao and Chen, Hanning and Wei, Yong and Chen, Minghan},
  year         = {2023},
  howpublished = {https://github.com/EnzeXu/ChemGNN_Dataset/raw/main/ChemGNN_Dataset.zip},
  publisher    = {GitHub},
  version      = {1.4.1}
}
```



# 3. Structure of the Repository


```
ChemGNN
┌── ChemGNN/
├────── models/
├────── utils/
├── data/
├── processed/
├── config.py.example
├── LICENSE
├── README.md
├── requirements.txt
└── run.py
```

- `ChemGNN/models/`: folder contains the model scripts
- `ChemGNN/utility/`: folder contains the utility scripts
- `data/`: folder contains the raw data (the destination of the `unzip` command in section 3.1)
- `processed/`: folder contains the processed data
- `config.py.example`: example file of a configuration to be applied
- `LICENSE`: license file
- `README.md`: readme file
- `requirements.txt`: main dependent packages (please follow section 3.1 to install all dependent packages)
- `run.py`: training script



# 4. Getting Started

This project is developed using Python 3.9 and is compatible with macOS, Linux, and Windows operating systems.

## 4.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone https://github.com/chenm19/ChemGNN.git
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



