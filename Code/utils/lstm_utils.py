import time

import numpy as np
import torch
import torch.nn as nn

from Code.models.lstm_model import EnforcedLSTMModel
from Code.utils.utils import get_data, split_on_change, assert_test_data, predict, make_report


def create_param_dict(
        model_name: str,
        input_size: int,
        output_size: int,
        n_layers: int,
        units: int,
        timespan: int,
        batch_size: int,
        val_range: int,
        epochs: int,
        lr: float,
        loss_function_name: str,

):
    """
    Create a parameter dictionary.

    :param model_name: Name of the model.
    :param input_size: Input dimension.
    :param output_size: Output dimension of the model.
    :param n_layers: Number of hidden layers.
    :param units: Number of neurons.
    :param timespan: Number of time steps to predict.
    :param batch_size: Batch size.
    :param val_range: Validation range.
    :param epochs: Number of epochs.
    :param lr: Learning rate.
    :param loss_function_name: Name of the loss function

    :return: Parameter dictionary.
    """
    param_dict = {
        "Model Name": model_name,
        "Architecture": "LSTM",
        "Input Dim": input_size,
        "Output Dim": output_size,
        "Number of Layers": n_layers,
        "Units": units,
        "Timespan": timespan,
        "Batch Size": batch_size,
        "Val Range": val_range,
        "Epochs": epochs,
        "Learning Rate": lr,
        "Loss": loss_function_name,
    }

    return param_dict


def initialize_lstm(
        model_name: str,
        input_size: int,
        output_size: int,
        n_layers: int,
        units: int,
        timespan: int,
        batch_size: int,
        val_range: int,
        epochs: int,
        lr: float,

) -> tuple[nn.Module, dict]:
    """
    Initialized the LSTM model.

    :param model_name: Name of the model.
    :param input_size: Input dimension.
    :param output_size: Output dimension of the model.
    :param n_layers: Number of hidden layers.
    :param units: Number of neurons.
    :param timespan: Number of time steps to predict.
    :param batch_size: Batch size.
    :param val_range: Validation range.
    :param epochs: Number of epochs.
    :param lr: Learning rate.

    :return: Initialized LSTM and parameter dict
    """
    param_dict = create_param_dict(
        model_name=model_name,
        input_size=input_size,
        output_size=output_size,
        n_layers=n_layers,
        units=units,
        timespan=timespan,
        batch_size=batch_size,
        val_range=val_range,
        epochs=epochs,
        lr=lr,
        loss_function_name="RMSE"
    )

    # Initialize LSTM
    lstm_model = EnforcedLSTMModel(
        input_size=param_dict["Input Dim"],
        hidden_size=param_dict["Units"],
        output_size=param_dict["Output Dim"],
        num_layers=param_dict["Number of Layers"]
    )

    return lstm_model, param_dict


def evaluate_lstm(param_dict: dict) -> None:
    """
    Evaluated the LSTM model and creates a report.

    :param param_dict: Parameter dictionary.
    """
    model = EnforcedLSTMModel(
        input_size=param_dict["Input Dim"],
        hidden_size=param_dict["Units"],
        output_size=param_dict["Output Dim"],
        num_layers=param_dict["Number of Layers"]
    )
    model.load_state_dict(torch.load(f"{param_dict["Model Name"]}.pth", weights_only=True))
    val_model = EnforcedLSTMModel(
        input_size=param_dict["Input Dim"],
        hidden_size=param_dict["Units"],
        output_size=param_dict["Output Dim"],
        num_layers=param_dict["Number of Layers"]
    )
    val_model.load_state_dict(torch.load(f"val_{param_dict["Model Name"]}.pth", weights_only=True))

    train_data = get_data("../Datasets/DORA_Train.csv")[:, 1:4]
    test_data = get_data("../Datasets/DORA_Test.csv")[:, 1:5]
    splitted_test = split_on_change(test_data)
    x_test, forcing, y_test = assert_test_data(train_data, splitted_test)

    predictions, val_predictions = predict(
        model=model, forcing=forcing, x_test=x_test, t=2500, val_model=val_model
    )

    losses = np.loadtxt(f"losses_{param_dict["Model Name"]}.txt")

    make_report(
        model_name=param_dict["Model Name"],
        test_data=y_test.numpy(),
        prediction=predictions,
        param_dict=param_dict,
        plot_range=300,
        losses=losses[:, 0],
        val_losses=losses[:, 1],
        val_prediction=val_predictions,
    )


def train_lstm(
        lstm_model: nn.Module,
        model_name: str,
        epochs: int,
        timespan: int,
        data_loader,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
        train_tensor: torch.Tensor,
        val_range: int,
        param_dict: dict
) -> None:
    """
    Training loop for the LSTM.

    :param lstm_model: Initialized LSTM model.
    :param model_name: Model name.
    :param epochs: Number of epochs.
    :param timespan: Number of time steps to predict.
    :param data_loader: DataLoader.
    :param optimizer: Optimizer.
    :param loss_function: Loss function.
    :param train_tensor: Tensor with training data.
    :param val_range: Validation range
    :param param_dict: Dictionary with parameters
    """
    losses = []
    val_losses = []
    times = []
    tpe = []
    best_loss = 1e20
    best_epoch = 0
    best_val_loss = 1e20
    best_val_epoch = 0

    start_time = time.time()

    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0
        epoch_time = time.time()

        for batch in data_loader:
            x0_train, targets, forcing = batch

            optimizer.zero_grad()
            output = lstm_model(inputs=x0_train, forcing=forcing, timespan=timespan)
            loss = torch.sqrt(loss_function(output, targets))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        tpe.append(time.time() - epoch_time)

        losses.append(epoch_loss / len(data_loader))
        epoch_loss = epoch_loss / len(data_loader)

        lstm_model.eval()
        with torch.no_grad():
            val1 = lstm_model(
                inputs=train_tensor[0, :-1].view(1, 2),
                forcing=train_tensor[:val_range, -1].view(-1, 1).unsqueeze(1),
                timespan=val_range,
            ).squeeze(1)
            val2 = lstm_model(
                inputs=train_tensor[2500, :-1].view(1, 2),
                forcing=train_tensor[2500: 2500 + val_range, -1]
                .view(-1, 1)
                .unsqueeze(1),
                timespan=val_range,
            ).squeeze(1)

        val_loss1 = torch.sqrt(loss_function(val1, train_tensor[:val_range, :-1]))
        val_loss2 = torch.sqrt(loss_function(val2, train_tensor[2500: 2500 + val_range, :-1], ))
        val_loss = (val_loss1.item() + val_loss2.item()) / 2
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(lstm_model.state_dict(), f"val_{model_name}.pth")
            print(f"New Best Val Loss: {best_val_loss}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(lstm_model.state_dict(), f"{model_name}.pth")
            print(f"New Best Loss: {epoch_loss}")

        e_time = time.time() - epoch_time
        times.append(e_time)

        print(
            f"Epoch {epoch},"
            f" Loss: {epoch_loss:.5f},"
            f" Val Loss: {val_loss:.5f},"
            f" Best Loss: {best_loss:.5f},"
            f" Best Epoch: {best_epoch},"
            f" Best Val Loss: {best_val_loss:.5f},"
            f" Best Val Epoch: {best_val_epoch},"
            f" Time: {e_time}"
        )

    comp_time = time.time() - start_time

    np.savetxt(
        f"losses_{model_name}.txt",
        np.stack((np.array(losses), np.array(val_losses)), axis=1),
    )

    param_dict["Best Loss"] = best_loss
    param_dict["Best Val Loss"] = best_val_loss
    param_dict["Best Epoch"] = best_epoch
    param_dict["Best Val Epoch"] = best_val_epoch

    if comp_time:
        param_dict["Duration"] = comp_time
        param_dict["Average Epoch Time"] = np.sum(times) / len(times)
        param_dict["TpE"] = np.sum(tpe) / len(tpe)
