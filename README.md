# Efficiently-Capturing-Causality-in-Data-with-Liquid-Time-Constant-Neural-Networks

This repository provides all necessary models, datasets and utilities to reproduce the results from the Master's Thesis `Efficiently Capturing Causality in Data with Liquid Time-Constant Neural Networks`. The performance of the Liquid Time-Constant Neural Network was evaluated and compared to other models using a dataset derived from the Duffing oscillator. Available models are:

| Models                                   | References                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| Liquid Time-Constant Neural Network      | [https://arxiv.org/abs/2006.04439](https://arxiv.org/abs/2006.04439)       |
| Neural ODE                               | [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)       |
| LSTM                                     | [https://www.bioinf.jku.at/publications/older/2604.pdf](https://www.bioinf.jku.at/publications/older/2604.pdf) |


## Prerequisites
All models were implemented with PyTorch 2.5.1 and Python 3.12. To use the Neural ODE, the Python package `torchdiffeq` from [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) is required. To install the `torchdiffeq` package:
```bash
pip install torchdiffeq
```


## Models


## Training and evaluating the models
