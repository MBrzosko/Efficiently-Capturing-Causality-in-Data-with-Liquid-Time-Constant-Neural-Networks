import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages


class DataLoader:
    def __init__(self, x, num_times_per_obs, mini_batch_size):
        self.x = x
        self.num_timesteps = x.shape[0]
        self.num_times_per_obs = num_times_per_obs
        self.mini_batch_size = mini_batch_size
        self.indices = np.arange(self.num_timesteps - num_times_per_obs + 1)
        self.current_index = 0
        np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            self.current_index = 0
            np.random.shuffle(self.indices)
            raise StopIteration

        s = self.indices[self.current_index : self.current_index + self.mini_batch_size]
        self.current_index += self.mini_batch_size

        x0_train = self.x[s, :]

        targets = np.zeros((self.num_times_per_obs, len(s), self.x.shape[1]))

        for i, start_index in enumerate(s):
            targets[:, i, :] = self.x[
                start_index : start_index + self.num_times_per_obs, :
            ]

        return (
            torch.tensor(x0_train, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.mini_batch_size))


def assert_test_data(train_data, splitted_test_data, seq=None):
    train_data = torch.tensor(train_data, dtype=torch.float32)
    splitted_test_data = torch.tensor(splitted_test_data, dtype=torch.float32)

    if seq is None:
        x_test = torch.zeros(7, 2)
        x_test[0] = splitted_test_data[0][0, :-1]
        x_test[1] = splitted_test_data[1][0, :-1]
        x_test[2] = train_data[0, :-1]
        x_test[3] = splitted_test_data[2][0, :-1]
        x_test[4] = train_data[2500, :-1]
        x_test[5] = splitted_test_data[3][0, :-1]
        x_test[6] = splitted_test_data[4][0, :-1]

        forcings = torch.zeros(7, 2500, 1)
        forcings[0] = splitted_test_data[0, :, -1].view(-1, 1)
        forcings[1] = splitted_test_data[1, :, -1].view(-1, 1)
        forcings[2] = train_data[0:2500, -1].view(-1, 1)
        forcings[3] = splitted_test_data[2, :, -1].view(-1, 1)
        forcings[4] = train_data[2500:, -1].view(-1, 1)
        forcings[5] = splitted_test_data[3, :, -1].view(-1, 1)
        forcings[6] = splitted_test_data[4, :, -1].view(-1, 1)

        y_test = torch.zeros(7, 2500, 2)
        y_test[0] = splitted_test_data[0, :, :-1]
        y_test[1] = splitted_test_data[1, :, :-1]
        y_test[2] = train_data[0:2500, :-1]
        y_test[3] = splitted_test_data[2, :, :-1]
        y_test[4] = train_data[2500:, :-1]
        y_test[5] = splitted_test_data[3, :, :-1]
        y_test[6] = splitted_test_data[4, :, :-1]

        return x_test, forcings, y_test

    else:
        x_test = torch.zeros(7, seq, 3)
        x_test[0] = splitted_test_data[0][0:seq, :]
        x_test[1] = splitted_test_data[1][0:seq, :]
        x_test[2] = train_data[0:seq, :]
        x_test[3] = splitted_test_data[2][0:seq, :]
        x_test[4] = train_data[2500 : 2500 + seq, :]
        x_test[5] = splitted_test_data[3][0:seq, :]
        x_test[6] = splitted_test_data[4][0:seq, :]

        y_test = torch.zeros(7, 2500, 3)
        y_test[0] = splitted_test_data[0]
        y_test[1] = splitted_test_data[1]
        y_test[2] = train_data[0:2500, :]
        y_test[3] = splitted_test_data[2]
        y_test[4] = train_data[2500:, :]
        y_test[5] = splitted_test_data[3]
        y_test[6] = splitted_test_data[4]

        return x_test, y_test


class FullDataLoader:
    def __init__(self, x, num_times_per_obs, mini_batch_size):
        self.x = x
        self.num_timesteps = x.shape[0]
        self.num_times_per_obs = num_times_per_obs
        self.mini_batch_size = mini_batch_size
        self.half_timesteps = self.num_timesteps // 2

        self.indices1 = np.arange(self.half_timesteps - num_times_per_obs + 1)
        self.indices2 = np.arange(
            self.half_timesteps, self.num_timesteps - num_times_per_obs + 1
        )

        self.current_index1 = 0
        self.current_index2 = 0

        np.random.shuffle(self.indices1)
        np.random.shuffle(self.indices2)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index1 >= len(self.indices1) and self.current_index2 >= len(
            self.indices2
        ):
            self.current_index1 = 0
            self.current_index2 = 0
            np.random.shuffle(self.indices1)
            np.random.shuffle(self.indices2)
            raise StopIteration

        half_batch_size = self.mini_batch_size // 2

        s1 = self.indices1[self.current_index1 : self.current_index1 + half_batch_size]
        s2 = self.indices2[self.current_index2 : self.current_index2 + half_batch_size]

        self.current_index1 += half_batch_size
        self.current_index2 += half_batch_size

        s = np.concatenate((s1, s2))
        np.random.shuffle(s)  # Shuffle to mix datapoints from both halves

        x0_train = self.x[s, :-1]

        targets = np.zeros((self.num_times_per_obs, len(s), self.x.shape[1] - 1))
        forcings = np.zeros((self.num_times_per_obs, len(s), 1))

        for i, start_index in enumerate(s):
            targets[:, i, :] = self.x[
                start_index : start_index + self.num_times_per_obs, :-1
            ]
            forcings[:, i, :] = self.x[
                start_index : start_index + self.num_times_per_obs, -1
            ].reshape(-1, 1)

        return (
            torch.tensor(x0_train, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(forcings, dtype=torch.float32),
        )

    def __len__(self):
        return int(np.ceil(len(self.indices1) / (self.mini_batch_size // 2)))


def create_batch(num_timesteps, num_times_per_obs, mini_batch_size, x):
    s = np.random.permutation(num_timesteps - num_times_per_obs + 1)[:mini_batch_size]

    x0_train = x[s, :]

    targets = np.zeros((num_times_per_obs, mini_batch_size, 3))

    for i, start_index in enumerate(s):
        targets[:, i, :] = x[start_index : start_index + num_times_per_obs, :]

    return torch.tensor(x0_train, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )


def create_full_batch(num_times_per_obs, mini_batch_size, x):
    if x.shape[0] > 2500:

        i1 = np.arange(0, 2500 - num_times_per_obs)
        i2 = np.arange(2500, 5000 - num_times_per_obs)
        indices = np.hstack((i1, i2))
        perm_indices = np.random.permutation(indices)[:mini_batch_size]

        x_full_train = x[perm_indices, :]

        targets = np.zeros((num_times_per_obs, mini_batch_size, x.shape[1]))
        for i, start_index in enumerate(perm_indices):
            targets[:, i, :] = x[start_index : start_index + num_times_per_obs, :]

        return torch.tensor(x_full_train, dtype=torch.float32), torch.tensor(
            targets, dtype=torch.float32
        )

    else:
        print(ValueError("Please use the create_batch function if only one time series is used."))


def predict(model, x_test, forcing, t, val_model=None):
    model.eval()
    predictions = torch.empty((x_test.shape[0], t, x_test.shape[1]))
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            pred = model(
                inputs=x_test[i].view(1, 2),
                forcing=forcing[i, :t].view(-1, 1).unsqueeze(1),
                timespan=t,
            )
            predictions[i] = pred.squeeze(1)

    if val_model:
        val_predictions = torch.empty((x_test.shape[0], t, x_test.shape[1]))
        val_model.eval()
        with torch.no_grad():
            for i in range(x_test.shape[0]):
                val_pred = val_model(
                    inputs=x_test[i].view(1, 2),
                    forcing=forcing[i, :t].view(-1, 1).unsqueeze(1),
                    timespan=t,
                )
                val_predictions[i] = val_pred.squeeze(1)

        return predictions, val_predictions

    return predictions


def create_full_plots(test_data, predictions, plot_range, val_predictions):
    plots = []
    freq = [0.2, 0.35, 0.46, 0.48, 0.49, 0.58, 0.75]
    for i in range(test_data.shape[0]):
        fig, ax = plt.subplots(2, sharex="col", figsize=(8.27, 11.69))
        criterion = nn.MSELoss()
        rmse = torch.sqrt(
            criterion(
                torch.tensor(test_data[i], dtype=torch.float32),
                torch.tensor(predictions[i], dtype=torch.float32),
            )
        )
        if val_predictions is not None:
            val_rmse = torch.sqrt(
                criterion(
                    torch.tensor(test_data[i], dtype=torch.float32),
                    torch.tensor(val_predictions[i], dtype=torch.float32),
                )
            )

            fig.suptitle(
                f"fa:{freq[i]}, RMSE: {rmse:.5f}, Val_RMSE: {val_rmse:.5f}",
                fontsize=16,
            )
        else:
            fig.suptitle(
                f"fa:{freq[i]}, RMSE: {rmse:.5f}",
                fontsize=16,
            )

        ax[0].plot(test_data[i, :plot_range, 0], label="True Values", marker="o")
        ax[0].plot(predictions[i, :plot_range, 0], label="Predictions", marker="x")
        ax[0].plot(
            val_predictions[i, :plot_range, 0], label="Val Predictions", marker="x"
        )
        ax[0].set(title=f"q1")
        ax[0].set_ylabel("q_1")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(test_data[i, :plot_range, 1], label="True Values", marker="o")
        ax[1].plot(predictions[i, :plot_range, 1], label="Predictions", marker="x")
        ax[1].plot(
            val_predictions[i, :plot_range, 1], label="Val Predictions", marker="x"
        )
        ax[1].set(title=f"q2")
        ax[1].set_ylabel("q_2")
        ax[1].grid(True)
        ax[1].legend()

        fig.tight_layout()

        plots.append(fig)

    return plots


def make_report(
    model_name: str,
    param_dict: dict,
    prediction: np.ndarray,
    test_data: np.ndarray,
    plot_range: int,
    losses=None,
    val_losses=None,
    val_prediction=None,
):
    plt.figure()
    plt.axis("off")
    text = "\n".join([f"{key}: {value}" for key, value in param_dict.items()])
    plt.text(
        0.1,
        0.5,
        text,
        ha="left",
        va="center",
        fontsize=12,
        transform=plt.gca().transAxes,
    )
    dict_page = plt.gcf()

    if val_losses and losses:
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        loss_page = plt.gcf()

    elif losses:
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        loss_page = plt.gcf()

    plots = create_full_plots(
        test_data, prediction, plot_range, val_prediction
    )

    plt.figure()
    plt.plot(
        test_data[3][:, 0],
        test_data[3][:, 1],
        label="Truth",
    )
    plt.plot(prediction[3][:, 0], prediction[3][:, 1], label="Prediction")
    plt.plot(val_prediction[3][:, 0], val_prediction[3][:, 1], label="Val Prediction")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.legend()
    plt.grid(True)
    q1_q2 = plt.gcf()

    loss_page = None

    with PdfPages(f"../evaluation/{model_name}_eval.pdf") as pdf:
        pdf.savefig(dict_page)
        if losses or val_losses:
            pdf.savefig(loss_page)
        pdf.savefig(q1_q2)

        for i in range(len(plots)):
            pdf.savefig(plots[i])


def get_data(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    return data


def get_change_indices(arr):
    fifth_feature = arr[:, -1]

    change_indices = np.where(fifth_feature[:-1] != fifth_feature[1:])[0] + 1

    return change_indices


def split_on_change(arr):
    change_indices = get_change_indices(arr)
    split_arrays = np.split(arr, change_indices)
    split_arrays = np.delete(split_arrays, -1, axis=-1)
    return split_arrays
