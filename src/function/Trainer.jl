module Trainer
include("../model/Model.jl")
include("./Optimiser.jl")
# --------------------
using ..Model: Perceptron
using ..Optimiser: perceptron_update!
# --------------------
export PerceptronTrainer, init_perceptron_trainer, train!

"""
  PerceptronTrainer(model::Perceptron, dataset::Tuple{Matrix{Float64},Vector{Int64}}, epochs::Int64, η::Float64)

A simple trainer for the perceptron model.

# Fields
    - `model::Perceptron`: Perceptron model to train.
    - `dataset::Tuple{Matrix{Float64},Vector{Int64}}`: Dataset for training.
    - `epochs::Int64`: Number of epochs for training.
    - `η::Float64`: Learning rate for training.

# Example
```julia
using .Trainer
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
```
"""
mutable struct PerceptronTrainer
    model::Perceptron
    dataset::Tuple{Matrix{Float64},Vector{Int64}}
    update_rule::Function
    epochs::Int64
    η::Float64
end

"""
  init_perceptron_trainer(model::Perceptron ,dataset::Tuple{Matrix{Float64},Vector{Int64}}, update_rule::Function=perceptron_update!, epochs::Int64=5, η::Float64=0.1) -> PerceptronTrainer

Initializes a trainer for the perceptron model.

# Arguments
    - `model::Perceptron`: Perceptron model to train.
    - `dataset::Tuple{Matrix{Float64},Vector{Int64}}`: Dataset for training.
    - `update_rule::Function`: Update rule for the model.
    - `epochs::Int64=5`: Number of epochs for training.
    - `η::Float64=0.1`: Learning rate for training.

# Returns
    - `PerceptronTrainer`: A trainer for the perceptron model.

# Example
```julia
using .Trainer
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
```
"""
function init_perceptron_trainer(
    model::Perceptron,
    dataset::Tuple{Matrix{Float64},Vector{Int64}},
    update_rule::Function=perceptron_update!
    ; epochs::Int64=100,
    η::Float64=0.1
)::PerceptronTrainer
    return PerceptronTrainer(model, dataset, update_rule, epochs, η)
end


"""
  train!(trainer::PerceptronTrainer) -> Perceptron

Train the perceptron model.

# Arguments
    - `trainer::PerceptronTrainer`: Trainer for the perceptron model.

# Returns

    - `Perceptron`: Trained perceptron model.

# Example
```julia
using .Trainer
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
model = train!(trainer)
```
"""
function train!(trainer::PerceptronTrainer)
    X, y = trainer.dataset
    for epoch in 1:trainer.epochs
        for i in eachindex(X[:, 1])
            x = X[i, :]
            target = y[i]
            trainer.model = trainer.update_rule(trainer.model, x, target, η=trainer.η)
        end
    end

    return trainer.model
end
end
