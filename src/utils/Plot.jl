module Plot

using Plots

export plot_dataset

"""
  plot_dataset(X::Matrix{Float64}, Y::Vector{Int64})

Plot the dataset.

# Arguments
- `X::Vector{Float64}`: Data.
- `Y::Vector{Float64}`: Labels.

# Example

```julia
using .Data
using .Plot

X, Y = generate_data()
plot_dataset(X, Y)
```
"""
function plot_dataset(
    X::Matrix{Float64},
    Y::Vector{Int64},
)
    p = scatter(
        X[:, 1],
        X[:, 2],
        group=Y,
        legend=:bottomleft,
        xlabel="x",
        ylabel="f(x)",
        title="Dataset"
    )

    savefig(p, "res/dataset/dataset.png")
end
end
