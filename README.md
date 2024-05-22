# Perceptron

## Description

Julia implementation of the perceptron algorithm from scratch for educational purposes. The perceptron, introduced by Frank Rosenblatt in 1958, is a fundamental binary classifier in machine learning. It employs a linear predictor function to categorize data into binary groups. The perceptron is pivotal in neural networks, demonstrating that machines can learn from data. Its simplicity and effectiveness have established it as a cornerstone in the study of neural computation and algorithmic learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Explanation](#mathematical-explanation)
- [License](#license)

## Installation

1. Clone the repository

```bash
git clone https://github.com/pierreaverty/perceptron.git
```

2. Change directory

```bash

cd perceptron
```

3. Install the dependencies

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Usage

```julia
using PerceptronJulia

# Generate the a random dataset, initialize the perceptron and the trainer
dataset = generate_data(n_samples=100)
perceptron = Perceptron(n_features=2)
trainer = Trainer(
    perceptron,
    dataset
)

# Train the perceptron
!train(trainer)

# Predict the class of a new data point
new_datapoint = [0.5, 0.5]

prediction = P(perceptron.W, X, perceptron.b)

println("The class of the new data point is: $prediction")
```

## Mathematical Explanation

The perceptron is a type of artificial neuron used in supervised learning for binary classifiers. It takes a vector of inputs, processes them linearly, and outputs a binary decision. Here’s a brief mathematical explanation:

### Input and Weights

Given an input vector \( \mathbf{x} = [x_1, x_2, \ldots, x_n] \) and a corresponding weight vector \( \mathbf{w} = [w_1, w_2, \ldots, w_n] \), the perceptron computes a weighted sum of the inputs:

\[ z = \mathbf{w} \cdot \mathbf{x} + b = \sum\_{i=1}^{n} w_i x_i + b \]

where \( b \) is the bias term.

### Activation Function

The perceptron uses a step function as the activation function to determine the output \( y \):

\[ y =
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
\]

This can also be expressed as:

\[ y = \text{sign}(z) \]

### Learning Rule

The perceptron learning algorithm adjusts the weights \( \mathbf{w} \) and bias \( b \) based on the prediction error. For each training sample \( (\mathbf{x}, t) \), where \( t \) is the target label, the weights and bias are updated as follows:

\[ \mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{w} \]
\[ b \leftarrow b + \Delta b \]

where the updates \( \Delta \mathbf{w} \) and \( \Delta b \) are given by:

\[ \Delta \mathbf{w} = \eta (t - y) \mathbf{x} \]
\[ \Delta b = \eta (t - y) \]

Here, \( \eta \) is the learning rate.

### Convergence

The perceptron algorithm iteratively adjusts the weights and bias until the classifier correctly classifies all the training samples or reaches a predefined number of iterations. For linearly separable data, the perceptron is guaranteed to converge to a solution.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
