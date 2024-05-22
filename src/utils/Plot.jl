module Plot
# --------------------
using Plots
# --------------------
export plot_dataset, plot_decision_boundary
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
        ylabel="y",
        title="Dataset"
    )

    savefig(p, "./val/dataset/dataset.png")
end

function plot_decision_boundary(
    W::Vector{Float64},
    b::Float64,
    X::Matrix{Float64},
    Y::Vector{Int64},
)
    p = scatter(
        X[:, 1],
        X[:, 2],
        group=Y,
        legend=:bottomleft,
        xlabel="x",
        ylabel="y",
        title="Decision Boundary"
    )

    x = 0:5
    y = (-W[1] * x .- b) ./ W[2]
    plot!(x, y, label="Decision Boundary", color=:red, lw=2)

    savefig(p, "./val/model/decision_boundary.png")
end
end
