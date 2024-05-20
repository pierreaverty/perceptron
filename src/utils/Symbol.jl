module Symbol

using LinearAlgebra

export ∑

"""
  ∑(x::Vector{Float64}, y::Vector{Float64}) -> Float64

Compute the dot product of two vectors.

# Arguments
- `x::Vector{Float64}`: First vector.
- `y::Vector{Float64}`: Second vector.

# Returns
- `Float64`: Dot product of the two vectors.

# Example

```julia
X = [1.0, 2.0, 3.0]
Y = [4.0, 5.0, 6.0]
dot_product = ∑(X, Y)
```
"""
function ∑(
    x::Vector{Float64},
    y::Vector{Float64}
)::Float64
    return dot(x, y)
end
end
