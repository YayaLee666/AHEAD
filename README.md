# AHEAD

This repository contains implementation for paper **Mitigating Anomaly Hallucination: A Model-Agnostic Framework for Unsupervised Anomaly Detection on Dynamic Graphs**.


## Experiment Reproduce

Code is written in Python and the proposed model is implemented using **Pytorch**.

To start training, use the following command:
```
python train.py --data_dir "your data dir" --dataset_name bitcoinotc --model_name AHEADGraphMixer
```

Make sure to adjust the '--processed_data' path to where your datasets are stored.


## Main Baseline Links

These following repositories represent the primary baseline models used in our experiments.

- SAD - https://github.com/D10Andy/SAD

- SLADE - https://github.com/jhsk777/SLADE

- GeneralDyG - https://github.com/YXNTU/GeneralDyG

## Extensibility

AHEAD is designed to be **model-agnostic**. The framework can be easily applied to other T-GNN backbones such as:
- GraphMixer - https://github.com/yule-BUAA/DyGLib
- DyGFormer - https://github.com/yule-BUAA/DyGLib
- TPNet  -  https://github.com/lxd99/TPNet
- BandRank  - https://github.com/YayaLee666/BandRank
- LSTEP  - https://github.com/kthrn22/L-STEP

Code for additional backbones will be released after paper acceptance, along with full experimental scripts and configurations.

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- NumPy  
- SciPy  
- tqdm  
- scikit-learn  
