"""
    module Model

This module defines a simple perceptron model for binary classification.

# Exported Types and Functions
- `Perceptron`: A struct representing the perceptron model.
- `init_perceptron`: A function to initialize the perceptron with random weights and bias.

# Example Usage
```julia
using .Model

# Initialize a perceptron with 2 input features
perceptron = init_perceptron(n=2)
println("Weights: ", perceptron.W)
println("Bias: ", perceptron.b)
"""
module Model

export Perceptron, init_perceptron

"""
    Perceptron(W::Vector{Float64}, b::Float64)

A simple perceptron model for classification.

# Fields
 - `W::Vector{Float64}`: Weight vector of the perceptron.
 - `b::Float64`: Bias of the perceptron.

# Example
```julia
W = [0.1, 0.2]
b = 0.3
perceptron = Perceptron(W, b)
```
"""
struct Perceptron
    W::Vector{Float64}
    b::Float64
end

"""
  init_perceptron(n::Int) -> Perceptron

Initializes a perceptron with random weights and bias.

# Arguments
- `n::Int`: Number of input features.

# Returns
- `Perceptron`: A perceptron with random weights and bias.

# Example

```julia
perceptron = init_perceptron(n=2)
```
"""
function init_perceptron(;
    n::Int
)::Perceptron
    W = randn(n)
    b = randn()

    return Perceptron(W, b)
end

end
