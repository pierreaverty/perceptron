module Activation

export θ

"""
    θ(z::Float64)

Heaviside step function, returns 1 if `z` is greater or equal to 0, otherwise returns 0.

# Fields
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
function θ(z::Float64)
    return z ≥ 0 ? 1 : 0
end
end
