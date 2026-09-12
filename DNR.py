import numpy as np
import gzip
import struct


class NeuralNetwork():

    def __init__(self):

        np.random.seed(1)

        #np.sqrt() scales down the random weights

        self.top_weights = (
            np.random.randn(252, 64)
            * np.sqrt(1 / 252)
        )

        self.middle_weights = (
            np.random.randn(280, 64)
            * np.sqrt(1 / 280)
        )

        self.bottom_weights = (
            np.random.randn(252, 64)
            * np.sqrt(1 / 252)
        )

        self.combined_weights = (
            np.random.randn(192, 128)
            * np.sqrt(1 / 192)
        )

        self.output_weights = (
            np.random.randn(128, 10)
            * np.sqrt(1 / 128)
        )

        self.top_biases = np.zeros((1, 64))
        self.middle_biases = np.zeros((1, 64))
        self.bottom_biases = np.zeros((1, 64))
        self.combined_bias = np.zeros((1, 128))
        self.output_bias = np.zeros((1, 10))

        self.learning_rate = 0.1


    def think(self, inputs):

        inputs = inputs.astype(float)

        images = inputs.reshape(-1, 28, 28)

        self.top_inputs = (
            images[:, 0:9, :]
            .reshape(-1, 252)
        )

        self.middle_inputs = (
            images[:, 9:19, :]
            .reshape(-1, 280)
        )

        self.bottom_inputs = (
            images[:, 19:28, :]
            .reshape(-1, 252)
        )

        self.z_top = (
            np.dot(
                self.top_inputs,
                self.top_weights
            )
            + self.top_biases
        )

        self.top_output = self.sigmoid(
            self.z_top
        )

        self.z_middle = (
            np.dot(
                self.middle_inputs,
                self.middle_weights
            )
            + self.middle_biases
        )

        self.middle_output = self.sigmoid(
            self.z_middle
        )

        self.z_bottom = (
            np.dot(
                self.bottom_inputs,
                self.bottom_weights
            )
            + self.bottom_biases
        )

        self.bottom_output = self.sigmoid(
            self.z_bottom
        )

        self.combined_output = np.concatenate(
            (
                self.top_output,
                self.middle_output,
                self.bottom_output
            ),
            axis=1
        )

        self.z_combined = (
            np.dot(
                self.combined_output,
                self.combined_weights
            )
            + self.combined_bias
        )

        self.hidden_output = self.sigmoid(
            self.z_combined
        )

        self.z_output = (
            np.dot(
                self.hidden_output,
                self.output_weights
            )
            + self.output_bias
        )

        self.final_output = self.softmax(
            self.z_output
        )

        return self.final_output


    def sigmoid(self, x):

        return 1 / (
            1 + np.exp(-x)
        )


    def sigmoid_derivative(self, x):

        return x * (1 - x)


    def softmax(self, x):

        exp_values = np.exp(
            x
            - np.max(
                x,
                axis=1,
                keepdims=True
            )
        )

        return (
            exp_values
            /
            np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )


    def train(
        self,
        training_inputs,
        training_outputs,
        epochs
    ):

        batch_size = 64

        for epoch in range(epochs):

            indices = np.random.permutation(
                len(training_inputs)
            )

            training_inputs = (
                training_inputs[indices]
            )

            training_outputs = (
                training_outputs[indices]
            )

            for start in range(
                0,
                len(training_inputs),
                batch_size
            ):

                end = start + batch_size

                x_batch = (
                    training_inputs[start:end]
                )

                y_batch = (
                    training_outputs[start:end]
                )

                output = self.think(
                    x_batch
                )

                output_delta = (
                    y_batch
                    - output
                )

                output_weights_adjustment = (
                    np.dot(
                        self.hidden_output.T,
                        output_delta
                    )
                    / len(x_batch)
                )

                output_bias_adjustment = np.mean(
                    output_delta,
                    axis=0,
                    keepdims=True
                )

                hidden_error = np.dot(
                    output_delta,
                    self.output_weights.T
                )

                hidden_delta = (
                    hidden_error
                    *
                    self.sigmoid_derivative(
                        self.hidden_output
                    )
                )

                combined_weights_adjustment = (
                    np.dot(
                        self.combined_output.T,
                        hidden_delta
                    )
                    / len(x_batch)
                )

                combined_bias_adjustment = np.mean(
                    hidden_delta,
                    axis=0,
                    keepdims=True
                )

                combined_error = np.dot(
                    hidden_delta,
                    self.combined_weights.T
                )

                top_error = (
                    combined_error[:, 0:64]
                )

                middle_error = (
                    combined_error[:, 64:128]
                )

                bottom_error = (
                    combined_error[:, 128:192]
                )

                top_delta = (
                    top_error
                    *
                    self.sigmoid_derivative(
                        self.top_output
                    )
                )

                middle_delta = (
                    middle_error
                    *
                    self.sigmoid_derivative(
                        self.middle_output
                    )
                )

                bottom_delta = (
                    bottom_error
                    *
                    self.sigmoid_derivative(
                        self.bottom_output
                    )
                )

                top_weights_adjustment = (
                    np.dot(
                        self.top_inputs.T,
                        top_delta
                    )
                    / len(x_batch)
                )

                middle_weights_adjustment = (
                    np.dot(
                        self.middle_inputs.T,
                        middle_delta
                    )
                    / len(x_batch)
                )

                bottom_weights_adjustment = (
                    np.dot(
                        self.bottom_inputs.T,
                        bottom_delta
                    )
                    / len(x_batch)
                )

                top_bias_adjustment = np.mean(
                    top_delta,
                    axis=0,
                    keepdims=True
                )

                middle_bias_adjustment = np.mean(
                    middle_delta,
                    axis=0,
                    keepdims=True
                )

                bottom_bias_adjustment = np.mean(
                    bottom_delta,
                    axis=0,
                    keepdims=True
                )

                self.top_weights += (
                    self.learning_rate
                    * top_weights_adjustment
                )

                self.middle_weights += (
                    self.learning_rate
                    * middle_weights_adjustment
                )

                self.bottom_weights += (
                    self.learning_rate
                    * bottom_weights_adjustment
                )

                self.combined_weights += (
                    self.learning_rate
                    * combined_weights_adjustment
                )

                self.output_weights += (
                    self.learning_rate
                    * output_weights_adjustment
                )

                self.top_biases += (
                    self.learning_rate
                    * top_bias_adjustment
                )

                self.middle_biases += (
                    self.learning_rate
                    * middle_bias_adjustment
                )

                self.bottom_biases += (
                    self.learning_rate
                    * bottom_bias_adjustment
                )

                self.combined_bias += (
                    self.learning_rate
                    * combined_bias_adjustment
                )

                self.output_bias += (
                    self.learning_rate
                    * output_bias_adjustment
                )

            predictions = self.think(
                training_inputs
            )

            epsilon = 1e-12

            loss = -np.mean(
                np.sum(
                    training_outputs
                    *
                    np.log(
                        predictions
                        + epsilon
                    ),
                    axis=1
                )
            )

            if epoch % 10 == 0:

                print(
                    "Epoch:",
                    epoch,
                    "Loss:",
                    loss
                )


def load_images(filename):

    with gzip.open(
        filename,
        "rb"
    ) as f:

        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return data.reshape(
            num_images,
            rows * cols
        )


def load_labels(filename):

    with gzip.open(
        filename,
        "rb"
    ) as f:

        magic, num_labels = struct.unpack(
            ">II",
            f.read(8)
        )

        return np.frombuffer(
            f.read(),
            dtype=np.uint8
        )


if __name__ == "__main__":

    x_train = load_images(
        "train-images-idx3-ubyte.gz"
    )

    y_train = load_labels(
        "train-labels-idx1-ubyte.gz"
    )

    x_test = load_images(
        "t10k-images-idx3-ubyte.gz"
    )

    y_test = load_labels(
        "t10k-labels-idx1-ubyte.gz"
    )

    print("Original shapes:")

    print(
        "x_train:",
        x_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "x_test:",
        x_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    x_train = (
        x_train.astype(float)
        / 255.0
    )

    x_test = (
        x_test.astype(float)
        / 255.0
    )

    y_train_onehot = (
        np.eye(10)[y_train]
    )

    y_test_onehot = (
        np.eye(10)[y_test]
    )

    print("\nPrepared shapes:")

    print(
        "x_train:",
        x_train.shape
    )

    print(
        "y_train_onehot:",
        y_train_onehot.shape
    )

    print(
        "x_test:",
        x_test.shape
    )

    print(
        "y_test_onehot:",
        y_test_onehot.shape
    )

    training_amount = 20000

    small_x_train = (
        x_train[:training_amount]
    )

    small_y_train = (
        y_train_onehot[:training_amount]
    )

    neural_network = NeuralNetwork()

    neural_network.train(
        small_x_train,
        small_y_train,
        100
    )

    predictions = neural_network.think(
        x_test
    )

    predicted_digits = np.argmax(
        predictions,
        axis=1
    )

    correct = np.sum(
        predicted_digits
        == y_test
    )

    accuracy = (
        correct
        / len(y_test)
    )

    print(
        "\nCorrect test predictions:",
        correct,
        "/",
        len(y_test)
    )

    print(
        "Test accuracy:",
        accuracy * 100,
        "%"
    )

    print("\nFirst 10:")

    for i in range(10):

        print(
            "Actual:",
            y_test[i],
            "| Predicted:",
            predicted_digits[i]
        )

    print(
        "\nTraining images used:",
        training_amount
    )
