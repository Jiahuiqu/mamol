# Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification

Official PyTorch implementation of the paper:  
**“Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification”**  
(arXiv: https://arxiv.org/abs/2511.11460)
<div align="center">
  <img src="fig/framework.jpg" width="720">
</div>
---

## Introduction

Remote sensing tasks often rely on **multiple sensor modalities** such as RGB, multispectral (MS), and SAR.  
However, in real-world deployments, some modalities may be **missing** due to sensor failure, occlusion, or environmental interference.

Prior multimodal models degrade significantly under such missing-modality conditions, or require training multiple separate models for each missing pattern — which is computationally inefficient.

To address these limitations, we propose an **efficient Mixture-of-Experts (MoE) framework** designed for modality-missing remote sensing classification.

### Key Contributions

- **Dynamic Modality-Aware Routing**  
  Selects appropriate experts depending on which modalities are available.

- **Shared + Modality-Specific Experts**  
  Preserves cross-modal generalization while enabling specialization.

- **Parameter-Efficient Training**  
  Only expert parameters are trained — the backbone remains frozen.

- **Unified Missing-Modality Learning**  
  A single model handles *all* missing cases (missing-HSI, missing-SAR, missing-MS, etc.).

---

## Usage
### Enviromen
  We follow environment by [DCP](https://github.com/hulianyuyy/Deep_Correlated_Prompting).

### Prepare Dataset for RGB
  We follow RGB and text seting by [DCP](https://github.com/hulianyuyy/Deep_Correlated_Prompting).
```
python make_arrow.py --dataset [DATASET] --root [YOUR_DATASET_ROOT]
```
### Prepare Dataset for remote
  The data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1nbOzUDTT0GXN8VDpw7ldWTt7WG-NRYG_?usp=sharing).
  ```txt
  datasets/
  ├── data_2013/
  │   ├── HSI.mat
  │   ├── LiDAR.mat
  │   ├── MSI.mat
  │   ├── All_label.mat
  ├── data_trento/
  │   ├── HSI.mat
  │   ├── LiDAR.mat         
  │   ├── All_label.mat
  ├── data_augsburg/
  │   ├── HSI.mat
  │   ├── LiDAR.mat     
  │   ├── SAR.mat       
  │   ├── All_label.mat
  └──  make_arrow.py
  ```
  
  ```
  python make_arrow.py
  ```

### Implementation
In mm-imdb and other datasets, the implementation all followed DCP! The key implementations for our proposed method are located in the [clip_missing_aware_moe_module.py](./mamol/modules/clip_missing_aware_moe_module.py) and [vision_transformer_moe.py](./mamol/modules/vision_transformer_moe.py), which defines the prompting approaches and base modules, respectively.

In remote sencing datasets, [clip_missing_aware_moe_module_remote.py](./mamol/modules/clip_missing_aware_moe_module_remote.py) and [vision_transformer_moe_remote.py](./mamol/modules/vision_transformer_moe_remote.py) are what your needings.

### Train
```
python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101 or task_finetune_hatememes> \
        exp_name=<EXP_NAME>
```
Example command:
```
python run.py with data_root=/path_to_mmimdb num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_mmimdb exp_name=exp_base
or
python run.py with data_root=/path_to_houston13 num_gpus=1 num_nodes=1 per_gpu_batchsize=128 remote exp_name=mamol
```
### Evaluation
```
python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101 or task_finetune_hatememes> \
        load_path=<MODEL_PATH> \
        exp_name=<EXP_NAME> \
        prompt_type=<PROMPT_TYPE> \
        test_ratio=<TEST_RATIO> \
        test_type=<TEST_TYPE> \
        test_only=True     
```
Example command:
```
python run.py with data_root=/path_to_mmimd num_gpus=1 num_nodes=1 per_gpu_batchsize=4 task_finetune_mmimdb load_path=/path_to_your_pretrained.ckpt test_only=True test_ratio=0.7 test_type=both exp_name=exp_test\
or
python run.py with data_root=/path_to_houston13 num_gpus=1 num_nodes=1 per_gpu_batchsize=500 remote load_path=/path_to_your_pretrained.ckpt test_only=True test_ratio=0.7 test_type=both exp_name=exp_test\
```
The `/path_to_your_pretrained.ckpt` could be the `.pt` file with prefix `epoch-` in the output folder.

## Acknowledgements
This code is based on [DCP](https://github.com/hulianyuyy/Deep_Correlated_Prompting), [ViLT](https://github.com/dandelin/ViLT.git), [CLIP](https://github.com/openai/CLIP) and [MMP](https://github.com/yilunlee/missing_aware_prompts). Many thanks for their contributions. 
