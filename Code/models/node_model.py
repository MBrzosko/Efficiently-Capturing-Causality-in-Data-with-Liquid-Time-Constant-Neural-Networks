import numpy as np
import torch
import torch.nn as nn
import torchdiffeq


class NeuralODEFunc(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
    ):
        super(NeuralODEFunc, self).__init__()

        self.net = None

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t, y):

        return self.net(y)


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, y0, t):

        sol = torchdiffeq.odeint(self.odefunc, y0, t, rtol=1e-5, atol=1e-7)
        return sol


class NeuralODE(nn.Module):
    def __init__(self, odefunc, input_dim, output_dim):
        super(NeuralODE, self).__init__()
        self.odeblock = ODEBlock(odefunc)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, y0, t):
        out = self.odeblock(y0, t)
        out = self.fc(out)
        return out


class EnforcedNeuralODE(nn.Module):
    def __init__(self, odefunc, input_dim, output_dim):
        super(EnforcedNeuralODE, self).__init__()
        self.odeblock = ODEBlock(odefunc)
        self.t = torch.tensor(np.array([0.1]), dtype=torch.float32)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, forcing, timespan):

        outputs = torch.empty((timespan, inputs.shape[0], inputs.shape[1]))
        outputs[0] = inputs
        for t in range(1, timespan):
            stacked_input = torch.cat((inputs, forcing[t - 1]), dim=1)

            out = self.odeblock(stacked_input, self.t)
            out = self.fc(out)
            inputs = torch.squeeze(out, dim=0)
            outputs[t, :, :] = out

        return outputs
