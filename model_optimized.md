# Project Name - Data Processing Documentation

## 1. Input Parameters

### 1.a Input File
- **All processed data sets, including random and optimized (`Random1_{2-21}water_newcutoff`, `Random2_{1-21}water_newcutoff`, `Random3_{2-21}water_newcutoff`, `Random4_{1-21}water_newcutoff`, `Random_1water_newcutoff`, `Optimized`, `Optimized_test`). This experiment uses the newcutoff data set.**
- **Format**: XML (.pt)
- **Example Data**: Random1_2water_newcutoff.pt
  ```Random1_2water_newcutoff.pt
  Data(x=[6006, 1262], edge_index=[2,13644], edge_attr=[13644], y=[1001], z=[6006,3])

### 1.b Input Hyperparameters File
- **Hyperparameters include: `the address to load the data set`, `the address to save the results`, `learning rate`, `batch size`, `epoch` and `seed` The current experient is epoch=3000(maybe shorter or longer), lr=0.003, batch_size=1024, seed=int(time.time())**
- **Format**: .py
- **Example**: config.py
  ```
  class Config:
    main_path = "./"
    dataset = "waterforce"
    root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIGS'.format(dataset)
    root_force = main_path + 'data/{}/AIFORCE'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "CONFIG_{}"
    format_force = "FORCE_{}"
    format_eigen = "ENERGY_{}"
    loss_fn_id = 1
    epoch = 3000
    epoch_step = 1  # print loss every {epoch_step} epochs
    batch_size= 1024
    lr = 0.003
    seed = int(time.time())
    weight = 0.0
### 1.c Input Model Architecture
- **The model architecture mainly modifies the number of layers of CEAL (`self.conv_num=2`) and the number of layers of pre_layer and post_layer inside CEAL (`pre_layers=3, post_layers=3`). The current experiment is Ceal=2, pre_layer=2, post_layer=2, dropout=0.0.**
- **Format**: .py
- **Example**: model_new.py
  ```
  class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_emb = Embedding(20, 10)  # self.edge_emb = Embedding(4, 50)
        self.conv_num = 2
        self.in_num = 1262
        aggregators = ['sum', 'mean', 'min', 'max', 'std']
        self.weights = torch.nn.Parameter(torch.rand(len(aggregators)))
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.std_weight = nn.Parameter(torch.rand(1))

        for _ in range(self.conv_num):
            if _ == 0:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            else:
                conv = CEALConv(in_channels=self.in_num, out_channels=self.in_num, weights=self.weights,
                                aggregators=aggregators, edge_dim=10, towers=1, pre_layers=3, post_layers=3,
                                divide_input=False)
                norms = BatchNorm(self.in_num)
            self.convs.append(conv)
            self.batch_norms.append(norms)

        self.mlp1 = Sequential(Linear(1262, 1262), ReLU())
        self.mlp2 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 1))
        self.mlp3 = Sequential(Linear(self.in_num, 600), ReLU(), Linear(600, 100), ReLU(), Linear(100, 3))
  
## 2. Output Results

### 2.a Run-time information in the screen
- `Epoch` `Loss_train` `Loss_val`  `Gradients` (The norm of the gradient of the last layer of energy mlp.weight) `Learning rate` `Duration of this epoch` `Total elapsed running time` `Total time remaining`
- **Example**:
  ```
  Epoch [01183/02000] Loss_train:0.000018 Loss_val:0.000010 Gradients:0.067064 Lr:0.000034 (Time:115.487673s Time total:2247.77min Time   
  remain: 1552.35min)
### 2.b Output Result File
- **The output `results.png` is the result picture displayed in the usual report. The output result.png is the result picture displayed in the usual report. All results are saved in `./train/waterforce`**

### 2.c Training Parameter Record File
- **Included: `gradients.png` `loss_combine.png`  `loss_last_half.png` `loss_last_quarter.png`  `lr.png`  `val_loss_whole.png` `loss_whole.png`. All results are saved in `./train/waterforce`**

### 2.d Test Result File of Test Set
- **Included: `test_pred.npy` `test_true.npy` The actual values ​​and predicted values ​​of the test set are saved respectively. All results are saved in `./train/waterforce`**

### 2.e Final Model File
- **Included: `final_state_dict.pth` This is the final saved model, which needs to be used for further test set testing. All results are saved in `./train/waterforce`**
