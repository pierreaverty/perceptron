module Model
include("../utils/Symbol.jl")
# --------------------
using .Symbol
# --------------------
export Perceptron, init_perceptron, P, ɸ, update!
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
using .Model

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

"""
  P(W::Vector{Float64}, X::Vector{Float64}, b::Float64) -> Int64

Predicts the output of a perceptron model.

# Arguments
- `W::Vector{Float64}`: Weight vector of the perceptron.
- `X::Vector{Float64}`: Input vector.
- `b::Float64`: Bias of the perceptron.

# Returns
- `Int64`: Output of the perceptron model.

# Example

```julia
using .Model

W = [0.1, 0.2]
X = [1.0, 2.0]
b = 0.3
output = P(W, X, b)
```
"""
function P(
    W::Vector{Float64},
    X::Vector{Float64},
    b::Float64
)::Int64
    return ɸ(∑(W, X) + b)
end

"""
  ɸ(x::Float64) -> Float64

Heaviside step function as an activation function.

# Arguments
- `x::Float64`: Input value.

# Returns
- `Float64`: Output value of the Heaviside step function.

# Example

```julia
using .Model

output = ɸ(0.5)
```
"""
function ɸ(
    x::Float64
)::Int64
    return x > 0 ? 1 : 0
end

"""
  update!(perceptron::Perceptron, X::Vector{Float64}, y::Int64; η::Float64=0.1) -> Perceptron

Updates the weights and bias of a perceptron model.

# Arguments
- `perceptron::Perceptron`: Perceptron model.
- `X::Vector{Float64}`: Input vector.
- `y::Int64`: True label.
- `η::Float64`: Learning rate.

# Returns
- `Perceptron`: Updated perceptron model.

# Example

```julia
using .Model

perceptron = init_perceptron(n=2)
X = [1.0, 2.0]
y = 1
perceptron = update!(perceptron, X, y)
```
"""
function update!(
    perceptron::Perceptron,
    X::Vector{Float64},
    y::Int64,
    ; η::Float64=0.1
)::Perceptron
    # Compute the prediction
    ŷ = P(perceptron.W, X, perceptron.b)

    # Compute the error
    δ = y - ŷ

    # Compute the update
    Δw = η * δ .* X
    Δb = η * δ

    # Update the weights and bias
    perceptron.W .+= Δw
    perceptron.b += Δb

    return perceptron
end

end
