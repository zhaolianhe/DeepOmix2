# DeepOmix2.0 Multi-modal data integration for pan-cancer clinical insights with explainable deep learning and large-scale pre-trained model 

<img src="graphical pipeline.png" width="625px" align="middle" />

### Description

### 

*In this work we propose to use omics and pathology images to build a clinical intepratation model for various tasks including survival analyisis， therapy responses，staging prediction and metastatic state. We present DeepOmix with a pretrained pathology model for pan-cancer and a widely application on various cancer types. Our model is tested on an held-out set and an external set of 789 WSIs from three institutions. *

© This code is made available for non-commercial academic purposes. 

#### Acknowledgment

We acknowledge all authors for DeepOmix2.(under MIT license).

### Installation  

First, we need to install the following packages:

* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia A00 x 8)
* Python (3.7.1), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), openslide (3.4.1), pandas (1.0.3), pillow (7.0.0), PyTorch (1.5.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.6).

Next, use the environment configuration file to create a conda environment:
```shell
conda env create -f env.yml
```

Activate the environment:
```shell
conda activate deepOmix_latest
```
install the package in the environment by running the following command:
```shell
pip install git+https://github.com/zhaolianhe/DeepOmix2.git
```

When done running experiments, to deactivate the environment:
```shell
conda deactivate deepOmix_latest
```
Please report any issues in the public forum.

### Examples

You may find a complete example of how to use DeepOmix in the examples:

  * Step1: Data Preparation

We chose to encode each pathology image patch with a 1024-dim feature vector using a pretrained MODEL( you can choose with Resnet18,Resnet50,Resnet101,ViT or our own in-house pre-trained model on baidu disk as follows (https://pan.baidu.com/s/1932eMehc0yrWgbXl2hbRSQ?pwd=pt9s pw: pt9s). For each pathology image, these features are expected to be saved as matrices of size N x 1024, where N is the number of patches from each whole slide image. The following folder structure is as follows. DATA_ROOT_DIR is the base directory of all datasets.
```bash
DATA_ROOT_DIR/
    └──DATASET_DIR/
         ├── image_files
                ├── slide_1.h5
                ├── slide_1.pt
                └── ...
         └── omics_files
                ├── patient_1_mutation.txt
                ├── patient_1_expression.txt
                └── ...
```


  * Step2: Training
    
We randomly partitioned our dataset into training, validation, and test splits. There is an example of 80/10/10 splits for the example dataset in the example file to evaluate the model's performance.

``` shell
CUDA_VISIBLE_DEVICES=0 python main_train.py  --task survival_prediction  --data_root_dir DATA_ROOT_DIR > log.data
```
The GPU to use for training can be specified using CUDA_VISIBLE_DEVICES, in the example command, GPU 0 is used. Other arguments such as drop_out, early_stopping, lr, and max_epochs can be specified with a parameter file.

By default results will be saved to **results/task** corresponding to the task input argument from the user. 

  * Step3: Prediction
    
User also has the option of using the evaluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:

``` shell
CUDA_VISIBLE_DEVICES=0 python prediction.py --task survival_analysis  --results_dir results --data_root_dir DATA_ROOT_DIR
```

  
### Issues
- Please report all issues on the public forum.

### References
[1] Zhao L, Dong Q, Luo C, Wu Y, Bu D, Qi X, Luo Y, Zhao Y. DeepOmix: A scalable and interpretable multi-omics deep learning framework and application in cancer survival analysis. Comput Struct Biotechnol J. 2021 May 1;19:2719-2725. doi: 10.1016/j.csbj.2021.04.067. PMID: 34093987; PMCID: PMC8131983.
