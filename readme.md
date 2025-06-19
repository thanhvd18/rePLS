# Residual Partial Least Squares Learning

This repository contains code, scripts, and resources for the study of Residual Partial Least Squares (rePLS) for statistical learning and neuroimaging analysis.


## Installation
```
pip install rePLS
```

## Data

- Data used in this project are from the following sources:
  - [ADNI (Alzheimer's Disease Neuroimaging Initiative)](https://adni.loni.usc.edu/)
  - [OASIS (Open Access Series of Imaging Studies)](https://www.oasis-brains.org/)

For visualize in brain space, you can use Matlab with [plotSurface](https://github.com/thanhvd18/plotSurface) or Python with [PySurfer](https://pysurfer.github.io/) .

## Usage
 - Run all experiments:
```bash
bash exps/run_all.sh
```
 - Generate a specific figure:
```bash
bash exps/run_figure1.sh
```


## Citing

If you use this code in your research, please cite the paper:

```
@article{chen2024residual,
  title={Residual Partial Least Squares Learning: Brain Cortical Thickness Simultaneously Predicts Eight Non-pairwise-correlated Behavioural and Disease Outcomes in Alzheimer’s Disease},
  author={Ch{\'e}n, Oliver Y and V{\~u}, Duy Thanh and Diaz, Christelle Schneuwly and Bodelet, Julien S and Phan, Huy and Allali, Gilles and Nguyen, Viet-Dung and Cao, Hengyi and He, Xingru and M{\"u}ller, Yannick and others},
  journal={bioRxiv},
  year={2024}
}
```