# Efficiently-Capturing-Causality-in-Data-with-Liquid-Time-Constant-Neural-Networks

This repository provides all necessary models, datasets and utilities to reproduce the results from the Master's Thesis `Efficiently Capturing Causality in Data with Liquid Time-Constant Neural Networks`. The performance of the Liquid Time-Constant Neural Network was evaluated and compared to other models using a dataset derived from the Duffing oscillator. Available models are:

| Models                                   | References                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| Liquid Time-Constant Neural Network      | [https://arxiv.org/abs/2006.04439](https://arxiv.org/abs/2006.04439)       |
| Neural ODE                               | [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)       |
| LSTM                                     | [https://www.bioinf.jku.at/publications/older/2604.pdf](https://www.bioinf.jku.at/publications/older/2604.pdf) |


## Requirements
All models were implemented with PyTorch 2.5.1 and Python 3.12. To use the Neural ODE, the Python package `torchdiffeq` from [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) is required. To install the `torchdiffeq` package:
```bash
pip install torchdiffeq
```

## Models

### Liquid Time-Constant Neural Network 
The LTC model can be found under `Code.models.ltc_model`. Two versions of the LTC model are provided. The difference between the `LTCModel` and the `EnforcedLTCModel` is that the enforced model requires the forcing at a given time step as an exogenous input, whereas the plain model uses the forcing as a feature input.
The `LTCCell` and `Wiring` were taken from [https://github.com/mlech26l/ncps](https://github.com/mlech26l/ncps).

### Neural ODE
The Neural ODE consists of two components. The `NeuralODEFunc` parametrises the derivative of the given input through a Feedforward Neural Network. The `ODEBlock` comprises the ODE solver. Both must be embedded in the `NeuralODE` or `EnforcedNeuralODE`.
The model was derived from [https://github.com/YuliaRubanova/latent_ode](https://github.com/YuliaRubanova/latent_ode).

### LSTM
The `lstm_models.py` script contains only the `EnforcedLSTMModel`.

## Training and evaluating the models
