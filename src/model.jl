"""
    Neuron

A single neuron model.

# Fields
- `W::Matrix{Float64}`: Weights matrix.
- `β::Float64`: Bias.
- `g::Function`: Activation function.

# Examples
```julia
julia> neuron = Neuron(randn(2, 1), 0.0, x -> x)
Neuron([0.0; 0.0], 0.0, var"#1#2"())
```
"""
mutable struct Neuron
    W::Matrix{Float64}
    β::Float64
    g::Function
end
