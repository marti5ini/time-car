# The importance of Time in Causal Algorithmic Recourse

This repository contains the implementation of the work [The Importance of Time in Causal Algorithmic Recourse](https://) in which 
we motivate **the need to integrate the temporal dimension into causal algorithmic recourse methods** to enhance recommendations’ **plausibility and reliability**. 


# Quick Start

You can utilize the following Jupyter Notebook to reproduce the results presented in the paper: 
```
[TIME-CAR](https://github.com/marti5ini/time-car/blob/master/time-car.ipynb)
```

# Setup

The code requires a python version >=3.9, as well as some libraries listed in requirements file. For some additional functionalities, more libraries are needed for these extra functions and options to become available. 

```
git clone https://github.com/marti5ini/time-car.git
cd time-car
```

Dependencies are listed in requirements.txt, a virtual environment is advised:

```
python3 -m venv ./venv # optional but recommended
pip install -r requirements.txt
```

# Citation

If you use `time-car` in your research, please cite our paper:

```
@article{beretta2023importance,
  title={The Importance of Time in Causal Algorithmic Recourse},
  author={Beretta, Isacco and Cinquini, Martina},
  journal={arXiv preprint arXiv:2306.05082},
  year={2023}
}
