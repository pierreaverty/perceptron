module Model

include("../function/Activation.jl")
include("../utils/Math.jl")
# --------------------
using .Activation
using .Math
# --------------------
export Perceptron, init_perceptron, predict
"""
    Perceptron(W::Vector{Float64}, b::Float64)

A simple perceptron model for classification.

# Fields
 - `W::Vector{Float64}`: Weight vector of the perceptron.
 - `b::Float64`: Bias of the perceptron.

# Example
```julia
using .Model

W = [0.1, 0.2]
b = 0.3
perceptron = Perceptron(W, b)
```
"""
mutable struct Perceptron
    W::Vector{Float64}
    b::Float64
    activation::Function
end

"""
  init_perceptron(n::Int, activation::Function=Activation.H) -> Perceptron

Initializes a perceptron with random weights and bias.

# Arguments
- `n::Int`: Number of input features.
- `activation::Function=Activation.H`: Activation function.

# Returns
- `Perceptron`: A perceptron with random weights and bias.

# Example

```julia
using .Model

perceptron = init_perceptron(n=2)
```
"""
function init_perceptron(;
    n::Int,
    activation::Function=Activation.H
)::Perceptron
    W = randn(n)
    b = randn()

    return Perceptron(W, b, activation)
end

"""
  predict(perceptron::Perceptron, X::Vector{Float64}) -> Int64

Predicts the output of a perceptron model.

# Arguments
- `perceptron::Perceptron`: Perceptron model.
- `X::Vector{Float64}`: Input vector.

# Returns
- `Int64`: Output of the perceptron model.

# Example

```julia
using .Model

perceptron = init_perceptron(n=2)
X = [1.0, 2.0]

output = predict(perceptron, X)
```
"""
function predict(
    perceptron::Perceptron,
    X::Vector{Float64},
)::Int64
    return perceptron.activation(∑(perceptron.W, X) + perceptron.b)
end
end
