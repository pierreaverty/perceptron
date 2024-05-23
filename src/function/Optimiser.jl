module Optimiser

include("../model/Model.jl")
# --------------------
using ..Model: Perceptron, predict
# --------------------
export perceptron_update!
"""
  perceptron_update!(perceptron::Perceptron, X::Vector{Float64}, y::Int64; η::Float64=0.1) -> Perceptron

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
using .Optimiser
using .Model

perceptron = init_perceptron(n=2)
X = [1.0, 2.0]
y = 1
perceptron = perceptron_update!(perceptron, X, y)
```
"""
function perceptron_update!(
    perceptron::Perceptron,
    X::Vector{Float64},
    y::Int64,
    ; η::Float64=0.1
)::Perceptron
    # Compute the prediction
    ŷ = predict(perceptron, X)

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
