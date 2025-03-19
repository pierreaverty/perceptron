"""
    Neuron

A single neuron model.

# Fields
- `W::Matrix{Float64}`: Weights matrix.
- `β::Float64`: Bias.
- `g::Function`: Activation function.

# Examples
```julia
julia> using Perceptron
julia> neuron = Neuron(randn(2, 1), 0.0, x -> x)
Neuron([0.0; 0.0], 0.0, x -> x)
```
"""
mutable struct Neuron
    W::Matrix{Float64}
    β::Float64
    g::Function
end

"""
    y(neuron::Neuron, x::Vector{Float64})

Compute the forward pass of the neuron given the input `x`.

# Arguments
- `neuron::Neuron`: The neuron instance.
- `x::Vector{Float64}`: The input vector.

# Returns
- `Float64`: The output of the neuron.

# Examples
```julia
julia> using Perceptron
julia> neuron = Neuron(randn(2, 1), 0.0, x -> x)
Neuron([0.0; 0.0], 0.0, x -> x)

julia> y(neuron, [1.0, 2.0])
3.0
```
"""
function y(neuron::Neuron, x::Vector{Float64})
    z = dot(neuron.W, x) + neuron.β

    return neuron.g(z)
end
