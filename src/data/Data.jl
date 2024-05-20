module Data

export generate_data

"""
generate_data(n_samples::Int64=100, σ::Float64=0.5, μ₁::Float64=1.0, μ₂::Float64=3.0) -> Tuple{Matrix{Float64}, Vector{Int64}}

Generate random data for a binary classification problem.

# Arguments
- `n_samples::Int64=100`: Number of samples in the dataset.
- `σ::Float64=0.5`: Standard deviation of the data.
- `μ₁::Float64=1.0`: Mean of the first class.
- `μ₂::Float64=3.0`: Mean of the second class.

# Returns
- `Tuple{Matrix{Float64}, Vector{Int64}}`: Tuple containing the data and the labels.

# Example

```julia
using .Data

X, Y = generate_data(n_samples=100)
```
"""
function generate_data(;
    n_samples::Int64=100,
    σ::Float64=0.5,
    μ₁::Float64=1.0,
    μ₂::Float64=3.0
)::Tuple{Matrix{Float64},Vector{Int64}}
    n_samples = n_samples ÷ 2

    x₁ = randn(n_samples) .* σ .+ μ₁
    y₁ = randn(n_samples) .* σ .+ μ₁
    label₁ = fill(0, n_samples)

    x₂ = randn(n_samples) .* σ .+ μ₂
    y₂ = randn(n_samples) .* σ .+ μ₂
    label₂ = fill(1, n_samples)

    coord_x = [x₁; x₂]
    coord_y = [y₁; y₂]

    X = hcat(coord_x, coord_y)

    Y = [label₁; label₂]

    return X, Y
end
end
