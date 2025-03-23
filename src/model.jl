using .Optimization

"""
    Neuron(dim::Tuple{Int, Int}, g::Function)

Create a new neuron instance.

# Arguments
- `dim::Tuple{Int, Int}`: Input and output dimensions.
- `g::Function`: Activation function.

# Returns
- `Neuron`: The created neuron instance.

# Examples
```julia
julia> using Perceptron
julia> neuron = Neuron((2, 1), x -> x)
Neuron((2, 1), x -> x, [0.0 0.0], [1.0;;])
```
"""
mutable struct Neuron
    dim::Tuple{Int, Int}
    g::Function
    W::Matrix{Float64}
    β::Matrix{Float64}

    function Neuron(dim::Tuple{Int, Int}, g::Function)::Neuron
        new(dim, g, zeros(Float64, dim[2], dim[1]), ones(Float64, dim[2], 1))
    end

    function Neuron(input_dim::Int, output_dim::Int, g::Function)::Neuron
        new((input_dim, output_dim), g, zeros(Float64, output_dim, input_dim), ones(Float64, output_dim, 1))
    end
end

"""
    y(neuron::Neuron, x::Matrix{Float64})::Matrix{Float64}

Compute the forward pass of the neuron given the input `x`.

# Arguments
- `neuron::Neuron`: The neuron instance.
- `x::Matrix{Float64}`: The input matrix.

# Returns
- `Matrix{Float64}`: The output of the neuron.

# Examples
```julia
julia> using Perceptron
julia> neuron = Neuron((2, 1), x -> x)
Neuron((2, 1), x -> x, [0.0 0.0], [1.0])
julia> y(neuron, [1.0 2.0])
[1.0;;]
```
"""
function y(neuron::Neuron, x::Matrix{Float64})::Matrix{Float64}
    z = neuron.W * x' .+ neuron.β

    return neuron.g.(z)
end
