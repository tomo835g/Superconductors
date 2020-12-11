# Introduction

The data and the code for ***''Deep-Learning Estimation of Band Gap with the Reading-Periodic-Table Method and Periodic Convolution Layer'' by Tomohiko Konno, Journal of the Physical Society of Japan (2020)***

The paper is [open access](https://doi.org/10.7566/JPSJ.89.124006), and Arxiv version is found [here](https://arxiv.org/abs/1912.05916).

# Condition
The data and the codes can be used under the condition that you cite the following two papers. Also see Licence.

```
@article{Konno2018DeepLM,
  title={Deep Learning Model for Finding New Superconductors},
  author={Tomohiko Konno and H. Kurokawa and F. Nabeshima and Y. Sakishita and Ryo Ogawa and I. Hosako and A. Maeda},
  journal={ArXiv},
  year={2018},
  volume={abs/1812.01995},
  }
```

``` 
@article{doi:10.7566/JPSJ.89.124006,
author = {Konno ,Tomohiko},
title = {Deep-Learning Estimation of Band Gap with the Reading-Periodic-Table Method and Periodic Convolution Layer},
journal = {Journal of the Physical Society of Japan},
volume = {89},number = {12},pages = {124006},year = {2020},
doi = {10.7566/JPSJ.89.124006},
URL = { https://doi.org/10.7566/JPSJ.89.124006},
eprint = {https://doi.org/10.7566/JPSJ.89.124006},
}
```



# Code

1. The code that transforms chemical formula like H<sub>2</sub>O into **reading periodic table type data format**.
   ```chemical_formula_to_reading_periodic_table.py```

An example
```
test_formula = 'H2He5'
reading_periodic_table = ReadingPeriodicTable(formula=test_formula)
reading_periodic_table_form_data = reading_periodic_table.formula_to_periodic_table()
print(reading_periodic_table_form_data)
>> must print 4*7*32 data.
formula_dict_form=reading_periodic_table.from_periodic_table_form_to_dict_form(reading_periodic_table_form_data)
print(formula_dict_form)
>> must print {'H':2,'He':5}
```

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


# Requirements

torch,pymatgen
