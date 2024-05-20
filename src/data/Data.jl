module Data

export generate_data

"""
generate_data(n_samples::Int64=100) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate random data for a binary classification problem.

# Arguments
- `n_samples::Int64=100`: Number of samples in the dataset.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple containing the data and the labels.

# Example

```julia
X, Y = generate_data(n_samples=100)
```
"""
function generate_data(
    n_samples::Int64=100
)
    n_samples = n_samples ÷ 2

    x₁ = randn(n_samples) .+ 1.0
    y₁ = randn(n_samples) .+ 1.0
    label₁ = fill(0, n_samples)

    x₂ = randn(n_samples) .+ 3.0
    y₂ = randn(n_samples) .+ 3.0
    label₂ = fill(1, n_samples)

    coord_x = [x₁; x₂]
    coord_y = [y₁; y₂]

    X = hcat(coord_x, coord_y)

    Y = [label₁; label₂]

    return X, Y
end
end
