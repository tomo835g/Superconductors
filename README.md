# Introduction

The code and the data used in ***''Deep-Learning Estimation of Band Gap with the Reading-Periodic-Table Method and Periodic Convolution Layer" by Tomohiko Konno, Journal of the Physical Society of Japan (2020)***

The paper is [open access](), and Arxiv version is found [here](https://arxiv.org/abs/1912.05916).

# Condition
The data and the codes can be used under the condition that you cite the following two papers. Also see Licence.

```
@article{Konno2018DeepLM,
  title={Deep Learning Model for Finding New Superconductors},
  author={Tomohiko Konno and H. Kurokawa and F. Nabeshima and Y. Sakishita and Ryo Ogawa and I. Hosako and A. Maeda},
  journal={ArXiv},
  year={2018},
  volume={abs/1812.01995}}
```

``` 
@article{Konno2019DeeplearningEO,
  title={Deep-learning estimation of band gap with the reading-periodic-table method and periodic convolution layer},
  author={Tomohiko Konno},
  journal={ArXiv},
  year={2019},
  volume={abs/1912.05916}}
```



# Code for the model

1. Transforms chemical formula like H2O into reading periodic table type data format.
   ```chemical_formula_to_reading_periodic_table.py```


2. The code for model (Pytorch)
    ```network_band_gap_estimation.py```
It also requires `periodic_shift_conv2D.py`

# Data


1. The data used for band gap binary classification.
    ```band_gap_data_binary_used.csv```
    The list of materials with band gap existence; 0 for no band gap, and 1 for band gap.
   
1. The data used for band gap value regression.
    ```band_gap_data_reg_used.csv```
    The list of materials and band gap values.



