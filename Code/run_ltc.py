import torch
import torch.nn as nn
import torch.optim as optim

from Code.utils.utils import get_data, FullDataLoader
from Code.utils.ltc_utils import initialize_ltc, train_ltc, evaluate_ltc


def main():
    # Model parameters
    units = 32
    input_size = 3
    wiring_output_size = 3
    output_size = 2
    timespan = 10

    # Dataset and optimization parameters
    ode_unfolds = 10
    epochs = 5
    batch_size = 512
    lr = 0.0001
    val_range = 500

    model_name = f"enforced_ltc_u{units}_ode{ode_unfolds}_ts{timespan}_e{epochs}_bs{batch_size}_lr{lr}"

    # Load and preprocess train and validation data
    train_data = get_data("../Datasets/DORA_Train.csv")[:, 1:4]
    train_tensor = torch.tensor(train_data, dtype=torch.float32)

    data_loader = FullDataLoader(train_data, timespan, batch_size)

    # Initialize LTC
    ltc_model, wiring, param_dict = initialize_ltc(
        model_name=model_name,
        input_size=input_size,
        output_size=output_size,
        wiring_output_size=wiring_output_size,
        units=units,
        ode_unfolds=ode_unfolds,
        timespan=timespan,
        batch_size=batch_size,
        val_range=val_range,
        epochs=epochs,
        lr=lr,
    )

    # Initialize loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(ltc_model.parameters(), lr=lr)

    # Train the LTC
    train_ltc(
        ltc_model=ltc_model,
        model_name=model_name,
        epochs=epochs,
        train_tensor=train_tensor,
        timespan=timespan,
        loss_function=loss_function,
        optimizer=optimizer,
        data_loader=data_loader,
        val_range=val_range,
        param_dict=param_dict
    )

    # Evaluate LTC and create report
    evaluate_ltc(param_dict=param_dict, wiring=wiring)


if __name__ == "__main__":
    main()
