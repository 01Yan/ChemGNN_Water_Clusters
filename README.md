# ChemGNN_Water_Clusters
Utilizing ChemGNN and SOAP to Predict the Properties of Water Clusters

## Getting Started

### Install Packages

(1) Install main dependent packages

`pip install -r requriements.txt`

(2) Install packages torch-scatter, torch-sparse, torch-cluster and torch-geometric manually corresponding to your operating systems and GPU version.

CUDA Example (If you are not using CUDA 11.3, please modify the suffix part "cuXXX" of each following url to match your CUDA version):

```
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-geometric
```

### Download Datasets
Download the processed 1-21-water data set in the following link:

`https://www.dropbox.com/home/MLDATA/MLDATA/1-21-water-dataset`

Unzip it, and replace the ‘dataset’ folder in the project with the obtained ‘dataset’ folder.
