import numpy as np
import torch
import torch.nn as nn
import torchdiffeq


"""
Installation of the python package torchdiffeq is required (https://github.com/rtqichen/torchdiffeq)
Additional information about the ODE code can be found here: https://github.com/YuliaRubanova/latent_ode
"""


class NeuralODEFunc(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
    ):
        """
        Feedforward neural network that parametrises the derivative of the input

        :param input_dim: Input dimension.
        :param hidden_dim: Number of neurons.
        :param output_dim:Output dimension.
        :param num_layers: Number of layers.
        """
        super(NeuralODEFunc, self).__init__()

        self.net = None

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NeuralODEFunc.

        :param y: Input.

        :return: Derivative of the input.
        """

        return self.net(y)


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        """
        Contains the ODE solvers. Takes the NeuralODEFunc as an attribute.

        :param odefunc: NeuralODEFunc.
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, y0: torch.Tensor, t: int) -> torch.Tensor:
        """
        ODE solver iteration.

        :param y0: Initial condition.
        :param t: Number of time steps.

        :return: Predicted sequence.
        """

        sol = torchdiffeq.odeint(self.odefunc, y0, t, rtol=1e-5, atol=1e-7)
        return sol


class NeuralODE(nn.Module):
    def __init__(self, odefunc: nn.Module, input_dim: int, output_dim: int):
        """
        Full NODE Model.

        :param odefunc: NeuralODEFunc.
        :param input_dim: Input dimension.
        :param output_dim: Output dimension.
        """
        super(NeuralODE, self).__init__()
        self.odeblock = ODEBlock(odefunc)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, y0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward pass of the NODE.

        :param y0: Initial condition.
        :param t: Number of time steps.

        :return: Predicted sequence
        """
        out = self.odeblock(y0, t)
        out = self.fc(out)
        return out


class EnforcedNeuralODE(nn.Module):
    def __init__(self, odefunc: nn.Module, input_dim: int, output_dim: int):
        """
        Full NODE Model with external forcing

        :param odefunc: NeuralODEFunc.
        :param input_dim: Input dimension.
        :param output_dim: Output dimension.
        """
        super(EnforcedNeuralODE, self).__init__()
        self.odeblock = ODEBlock(odefunc)
        self.t = torch.tensor(np.array([0.1]), dtype=torch.float32)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor, forcing: torch.Tensor, timespan: int) -> torch.Tensor:
        """
        Forward pass of the NODE with external enforcing.

        :param inputs: Initial condition.
        :param forcing: External forcing.
        :param timespan: Number of time steps.

        :return: Predicted sequence.
        """

        outputs = torch.empty((timespan, inputs.shape[0], inputs.shape[1]))
        outputs[0] = inputs
        for t in range(1, timespan):
            stacked_input = torch.cat((inputs, forcing[t - 1]), dim=1)

            out = self.odeblock(stacked_input, self.t)
            out = self.fc(out)
            inputs = torch.squeeze(out, dim=0)
            outputs[t, :, :] = out

        return outputs
