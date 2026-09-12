# Handwritten Digit Recognition Neural Network

## Project Overview

This project builds and trains a neural network **from scratch with Python and NumPy**. Its purpose is to recognize handwritten digits from **0 through 9** using the MNIST dataset.

Instead of using a machine-learning framework such as TensorFlow or PyTorch, the program manually performs the major steps of neural-network learning:

- loading and preparing image data;
- initializing weights and biases;
- performing forward propagation;
- calculating classification loss;
- performing backpropagation;
- updating the model's parameters; and
- measuring accuracy on images the network did not train on.

This makes the project useful for learning what happens inside a neural network.

## What Is MNIST?

MNIST is a dataset of grayscale images of handwritten digits. Each image:

- represents one digit from 0 to 9;
- is 28 pixels wide and 28 pixels tall;
- contains 784 pixels in total (`28 Ã— 28 = 784`); and
- stores each pixel's brightness as a number from 0 to 255.

The standard dataset contains 60,000 training images and 10,000 test images. This program deliberately trains on the first **20,000** training images and evaluates the trained model on all **10,000** test images.

## How the Project Fits Into Machine Learning

```text
Artificial Intelligence
â””â”€â”€ Machine Learning
    â””â”€â”€ Neural Networks
        â””â”€â”€ This Project: Handwritten-Digit Classification
```

This is a **supervised classification** project:

- **Supervised** means every training image comes with the correct answer, called a label.
- **Classification** means the model chooses among a fixed set of categories: digits 0â€“9.

## Network Architecture

The program does not send all 784 pixels through one initial layer. It divides every 28Ã—28 image into three horizontal regions:

| Region | Rows used | Input features | Neurons produced |
| --- | ---: | ---: | ---: |
| Top | 0â€“8 | `9 Ã— 28 = 252` | 64 |
| Middle | 9â€“18 | `10 Ã— 28 = 280` | 64 |
| Bottom | 19â€“27 | `9 Ã— 28 = 252` | 64 |

Each region is processed by its own fully connected layer. The three 64-value outputs are concatenated into 192 values. Those values pass through a 128-neuron hidden layer and finally into a 10-neuron output layer.

```text
28Ã—28 image
â”œâ”€â”€ Top 252 pixels â”€â”€â”€â”€â”€â†’ 64 neurons â”€â”€â”€â”€â”
â”œâ”€â”€ Middle 280 pixels â”€â”€â†’ 64 neurons â”€â”€â”€â”€â”¼â†’ 192 combined values
â””â”€â”€ Bottom 252 pixels â”€â”€â†’ 64 neurons â”€â”€â”€â”€â”˜
                                           â†“
                                    128 hidden neurons
                                           â†“
                                  10 output probabilities
                                           â†“
                              Predicted digit from 0 to 9
```

Although the image is divided into regions, this is **not a convolutional neural network (CNN)**. Every region uses a dense matrix multiplication, so it is better described as a custom multi-branch, fully connected neural network.

## Data Flow and Array Shapes

For a mini-batch of `B` images, the important array shapes are:

| Value | Shape | Meaning |
| --- | --- | --- |
| Input batch | `(B, 784)` | Flattened images |
| Reshaped images | `(B, 28, 28)` | Original image layout |
| Top input | `(B, 252)` | Top image pixels |
| Middle input | `(B, 280)` | Middle image pixels |
| Bottom input | `(B, 252)` | Bottom image pixels |
| Each regional output | `(B, 64)` | Features learned from one region |
| Combined regional output | `(B, 192)` | Three sets of 64 features |
| Hidden output | `(B, 128)` | Higher-level learned features |
| Final output | `(B, 10)` | Probability for every digit |

Understanding these shapes is important because matrix multiplication requires compatible dimensions.

## How a Neuron Works

A neuron first calculates a weighted sum:

```text
z = (input Ã— weight) + bias
```

- An **input** is information entering the neuron, such as a pixel value.
- A **weight** controls how strongly an input influences the neuron.
- A **bias** lets the neuron shift its response independently of the inputs.
- An **activation function** transforms the result and introduces nonlinearity.

Without nonlinear activation functions, several layers would behave like a single linear calculation and could not learn complex patterns effectively.

## Main Parts of the Code

### 1. Weight and bias initialization

The `NeuralNetwork.__init__()` method creates five weight matrices:

| Weight matrix | Shape | Connection |
| --- | --- | --- |
| `top_weights` | `(252, 64)` | Top pixels â†’ top neurons |
| `middle_weights` | `(280, 64)` | Middle pixels â†’ middle neurons |
| `bottom_weights` | `(252, 64)` | Bottom pixels â†’ bottom neurons |
| `combined_weights` | `(192, 128)` | Combined features â†’ hidden layer |
| `output_weights` | `(128, 10)` | Hidden layer â†’ digit scores |

Weights begin as small random numbers. Multiplication by `sqrt(1 / number_of_inputs)` controls their scale, helping prevent activations from becoming extremely large at the beginning of training.

Biases begin at zero. `np.random.seed(1)` makes the initial random values reproducible, so repeated runs begin from the same weights.

### 2. Forward propagation with `think()`

Forward propagation produces a prediction:

1. Convert the input values to floating-point numbers.
2. Reshape every 784-value row into a 28Ã—28 image.
3. divide each image into top, middle, and bottom regions.
4. Multiply each region by its corresponding weight matrix and add its biases.
5. Apply the sigmoid activation function to each regional result.
6. Concatenate the three regional outputs.
7. Pass the combined values through the 128-neuron hidden layer.
8. Produce 10 final scores and convert them into probabilities with softmax.

The final output might resemble:

```text
[0.01, 0.02, 0.03, 0.01, 0.04, 0.07, 0.02, 0.76, 0.03, 0.01]
```

The largest probability is at index 7, so the predicted digit is `7`.

### 3. Sigmoid activation

The hidden layers use:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

Sigmoid changes any real number into a value between 0 and 1. Its derivative is used during backpropagation to measure how sensitive a neuron's output is to change.

### 4. Softmax output

Softmax converts the 10 raw output scores into probabilities that add up to 1. Subtracting the largest score before exponentiation improves numerical stability and reduces the chance of overflow.

### 5. Mini-batch training

The `train()` method uses mini-batches of 64 images. An **epoch** is one complete pass through all selected training examples.

At the beginning of every epoch, the data is shuffled. The network then processes 64 examples at a time. Mini-batches provide a practical balance between one-example-at-a-time training and processing the complete dataset in one large calculation.

### 6. Backpropagation

Backpropagation determines how each weight and bias contributed to prediction error. The program moves backward through the network:

```text
Output error
    â†“
128-neuron hidden layer
    â†“
192 combined regional values
    â†“
Top, middle, and bottom branches
```

For the softmax output with cross-entropy loss, the output correction is represented as `correct_answer - prediction`. That correction is multiplied backward through the weight matrices. The sigmoid derivative is applied at hidden layers, and dot products calculate the average adjustments for the batch.

The model then updates each parameter using:

```text
parameter += learning_rate Ã— adjustment
```

Because this code defines the adjustment in the direction that reduces error, it uses `+=`. In the more common gradient notation, the gradient has the opposite sign and the update is written with `-=`.

### 7. Cross-entropy loss

Cross-entropy measures how well the predicted probabilities match the correct labels. A confident correct prediction gives a low loss; a confident incorrect prediction gives a high loss.

The small value `epsilon = 1e-12` prevents the program from attempting `log(0)`, which is undefined.

### 8. Testing and accuracy

After training, the program processes the test images. `np.argmax()` selects the index with the largest probability. That index is the predicted digit.

Accuracy is calculated as:

```text
accuracy = correct predictions / total test images
```

Test images are not used to update the weights. They estimate how well the model generalizes to previously unseen handwriting.

## Data Preparation

### Pixel normalization

Original pixel values range from 0 to 255. Dividing by 255 converts them to values between 0 and 1:

```python
x_train = x_train.astype(float) / 255.0
```

This keeps input values on a manageable scale and usually makes training more stable.

### One-hot encoding

The original label is one integer. The program converts it to a 10-value vector:

```text
Digit 3 â†’ [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

This format matches the network's 10 output probabilities.

## Required Files

Place the Python program and these four compressed MNIST files in the same directory:

```text
project-folder/
â”œâ”€â”€ neural_network.py
â”œâ”€â”€ train-images-idx3-ubyte.gz
â”œâ”€â”€ train-labels-idx1-ubyte.gz
â”œâ”€â”€ t10k-images-idx3-ubyte.gz
â””â”€â”€ t10k-labels-idx1-ubyte.gz
```

The filenames must match the strings used near the bottom of the Python program.

## Requirements

- Python 3
- NumPy

Install NumPy with:

```bash
python -m pip install numpy
```

The `gzip` and `struct` modules are included with Python and do not need separate installation.

## Running the Project

From the folder containing the code and MNIST files, run:

```bash
python neural_network.py
```

If the file still uses its uploaded name, run:

```bash
python "Pasted code(1).py"
```

The program prints:

- original and prepared dataset shapes;
- training loss every 10 epochs;
- the number of correct test predictions;
- test accuracy;
- actual and predicted values for the first 10 test images; and
- the number of training images used.

## Important Settings

| Setting | Current value | Effect |
| --- | ---: | --- |
| Learning rate | `0.1` | Size of each parameter update |
| Batch size | `64` | Images processed before each update |
| Epochs | `100` | Complete passes through training data |
| Training examples | `20,000` | Subset used from the training set |
| Output classes | `10` | Digits 0â€“9 |

A learning rate that is too high may make training unstable. One that is too low may make learning extremely slow. More epochs do not always guarantee better test performance because the model can eventually overfit its training data.

## What Makes This Project Interesting?

The top, middle, and bottom branches allow the model to initially learn from different regions independently. This can reflect useful digit structure: for example, the top of a `7`, middle of an `8`, and bottom of a `2` contain different visual clues. The 128-neuron layer later combines those regional clues to make a final decision.

However, splitting an image into fixed horizontal regions also creates limitations. The network does not explicitly understand nearby pixel patterns or handle shifted digits as naturally as a CNN would.

## Limitations

- It uses only 20,000 of the 60,000 available training images.
- It has no validation set for tuning settings independently of the test set.
- Sigmoid can suffer from vanishing gradients in deeper networks.
- It has no convolution, pooling, dropout, or data augmentation.
- Fixed horizontal splitting may be sensitive to where a digit appears in the image.
- It does not save trained weights, so every run trains from the beginning.
- It reports accuracy but does not show which digits are most frequently confused.

## Possible Improvements

- Train on all 60,000 training images.
- Create a validation set.
- Save and reload trained parameters.
- Replace sigmoid with ReLU in hidden layers.
- Add a confusion matrix and per-digit accuracy.
- Display misclassified images.
- Add early stopping to limit overfitting.
- Implement a CNN to learn local features such as edges and curves.
- Add command-line options for epochs, batch size, and learning rate.

## Summary

This project is a complete educational example of supervised machine learning. It receives handwritten-digit images, learns useful numerical patterns through forward propagation and backpropagation, and predicts one of 10 digit classes. Its most distinctive design choice is dividing every image into top, middle, and bottom branches before combining the learned features.

The project demonstrates the central idea of neural-network learning: **make a prediction, measure the error, send the correction backward, and adjust the parameters so the next prediction can improve.**
