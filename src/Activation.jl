module Activation

export θ

"""
    θ(z::Float64)::Int64

Heaviside step function, returns 1 if `z` is greater or equal to 0, otherwise returns 0.

# Arguments
- `z`: Input value

# Examples
```julia
julia> using Perceptron
julia> Activation.θ(0.0)
1
julia> Activation.θ(-0.1)
0
```
"""
θ(z::Float64)::Int64 = z ≥ 0 ? 1 : 0

end
