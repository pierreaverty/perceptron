module Optimization

export Optimizer, update

"""
    Optimizer

Abstract type representing an optimizer for training.
"""
abstract type Optimizer end

"""
    PerceptronOptimizer(η::Float64)

Create a new PerceptronOptimizer instance with learning rate `η`.

# Arguments
- `η::Float64`: The learning rate.

# Returns
- `optimizer::PerceptronOptimizer`: The optimizer instance.

# Examples
```julia
julia> using Optimization
julia> optimizer = PerceptronOptimizer(0.1)
PerceptronOptimizer(0.1)
```
"""
struct PerceptronOptimizer <: Optimizer
    η::Float64
end


"""
    update(optimizer::PerceptronOptimizer, W::Matrix{Float64}, β::Matrix{Float64}, D::Tuple{Matrix{Float64}, Matrix{Float64}}, ȳ::Matrix{Float64})::Tuple{Matrix{Float64},Matrix{Float64}}

Update the weights and biases of a perceptron using the Perceptron learning rule.

# Arguments
- `optimizer::PerceptronOptimizer`: The optimizer instance.
- `W::Matrix{Float64}`: The weight matrix.
- `β::Matrix{Float64}`: The bias matrix.
- `D::Tuple{Matrix{Float64}, Matrix{Float64}}`: The input data and target values.
- `ȳ::Matrix{Float64}`: The predicted values.

# Returns
- `ΔW::Matrix{Float64}`: The update for the weight matrix.
- `Δβ::Matrix{Float64}`: The update for the bias matrix.

# Examples
```julia
julia> using Optimization
julia> optimizer = PerceptronOptimizer(0.1)
PerceptronOptimizer(0.1)
julia> update(optimizer, rand(2, 2), rand(2, 2), (rand(2, 2), rand(2, 2)), rand(2, 2))
([0.1 0.1; 0.1 0.1], [0.1 0.1; 0.1 0.1])
```
"""
function update(
    optimizer::PerceptronOptimizer,
    W::Matrix{Float64},
    β::Matrix{Float64},
    D::Tuple{Matrix{Float64},
        Matrix{Float64}},
    ȳ::Matrix{Float64}
)::Tuple{Matrix{Float64},Matrix{Float64}}
    x, y = D
    ϵ = y .- ȳ

    ΔW = optimizer.η .* ϵ .* x
    Δβ = optimizer.η .* ϵ

    return W .+ ΔW, β .+ Δβ
end

end
