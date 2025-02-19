import torch
import torch.nn as nn


class EnforcedLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        LSTM Model with external enforcing.

        :param input_size: Input dimension.
        :param hidden_size: Number of neurons.
        :param num_layers: Number of layers.
        :param output_size: Output dimension.
        """
        super(EnforcedLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            inputs: torch.Tensor,
            forcing: torch.Tensor,
            timespan: int,
            h_state: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Foward pass of the LSTM Model.

        :param inputs: Initial condition.
        :param forcing: External forcing.
        :param timespan: Number of timesteps to predict.
        :param h_state: Hidden state.

        :return: Predicted sequence.
        """
        if h_state is None:
            h0 = torch.zeros(
                self.num_layers, inputs.shape[0], self.hidden_size, dtype=torch.float32
            )
            c0 = torch.zeros(
                self.num_layers, inputs.shape[0], self.hidden_size, dtype=torch.float32
            )
            h_state = (h0, c0)

        outputs = torch.empty((timespan, inputs.shape[0], inputs.shape[1]))
        outputs[0] = inputs
        for t in range(1, timespan):
            stacked_input = torch.cat((inputs, forcing[t - 1]), dim=1).unsqueeze(1)
            h_out, h_state = self.lstm.forward(stacked_input, h_state)
            h_out = self.fc(h_out).squeeze(1)
            inputs = h_out
            outputs[t, :, :] = h_out

        return outputs
