import torch
import torch.nn as nn
import torch.optim as optim

from Code.utils.utils import get_data, FullDataLoader
from Code.utils.node_utils import initialize_node, train_node, evaluate_node


def main():
    # Model parameters
    units = 32
    input_size = 3
    n_layers = 1
    output_size = 2
    timespan = 10

    # Dataset and optimization parameters
    ode_unfolds = 10
    epochs = 5
    batch_size = 512
    lr = 0.0001
    val_range = 500

    model_name = f"enforced_node_u{units}_ode{ode_unfolds}_ts{timespan}_e{epochs}_bs{batch_size}_lr{lr}"

    # Load and preprocess train and validation data
    train_data = get_data("../Datasets/DORA_Train.csv")[:, 1:4]
    train_tensor = torch.tensor(train_data, dtype=torch.float32)

    data_loader = FullDataLoader(train_data, timespan, batch_size)

    # Initialize NODE
    node_model, param_dict = initialize_node(
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
    )

    # Initialize loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(node_model.parameters(), lr=lr)

    # Train the NODE
    train_node(
        node_model=node_model,
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

    # Evaluate NODE and create report
    evaluate_node(param_dict=param_dict)


if __name__ == "__main__":
    main()
