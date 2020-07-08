## Description

`humpback` is Python package using `sklearn` API and provides tools for automated columns selection problem.

Main class is `ColumnsSelector` which selects best subset of columns according to provided heuristic and Information 
Criterion. See `example.py` for two examples of usage.

Package provides implementation of following Information Criterions: 
[AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion), 
[BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion), 
[mBIC](https://www.genetics.org/content/genetics/167/2/989.full.pdf),
[mBIC2](https://www.sciencedirect.com/science/article/abs/pii/S0167947311001459#!).

## Installation Guide
````
git clone https://github.com/qwymierne/humpback
pip install -r humpback/requirements.txt
pip install humpback
````