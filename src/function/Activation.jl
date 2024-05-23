module Activation

export H

"""
  H(x::Float64) -> Float64

Heaviside step function as an activation function.

# Arguments
- `x::Float64`: Input value.

# Returns
- `Float64`: Output value of the Heaviside step function.

# Example

```julia
using .ActivationJulia

output = H(0.5)
```
"""
function H(
    x::Float64
)::Int64
    return x > 0 ? 1 : 0
end
end
