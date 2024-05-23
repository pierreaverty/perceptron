module TrainerJulia
include("../model/Model.jl")
# --------------------
using ..Model: Perceptron, update!
# --------------------
export Trainer, init_perceptron_trainer, train!

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
using .TrainerJulia
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
```
"""
mutable struct Trainer
    model::Perceptron
    dataset::Tuple{Matrix{Float64},Vector{Int64}}
    epochs::Int64
    η::Float64
end

"""
  init_perceptron_trainer(model::Perceptron, dataset::Tuple{Matrix{Float64},Vector{Int64}}, epochs::Int64=5, η::Float64=0.1) -> PerceptronTrainer

Initializes a trainer for the perceptron model.

# Arguments
    - `model::Perceptron`: Perceptron model to train.
    - `dataset::Tuple{Matrix{Float64},Vector{Int64}}`: Dataset for training.
    - `epochs::Int64=5`: Number of epochs for training.
    - `η::Float64=0.1`: Learning rate for training.

# Returns
    - `PerceptronTrainer`: A trainer for the perceptron model.

# Example
```julia
using .TrainerJulia
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
```
"""
function init_perceptron_trainer(
    model::Perceptron,
    dataset::Tuple{Matrix{Float64},Vector{Int64}},
    ; epochs::Int64=100,
    η::Float64=0.1
)::Trainer
    return Trainer(model, dataset, epochs, η)
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
using .TrainerJulia
using .Model

model = init_perceptron(n=2)
dataset = generate_data()
trainer = init_perceptron_trainer(model, dataset)
model = train!(trainer)
```
"""
function train!(trainer::Trainer)
    X, y = trainer.dataset
    for epoch in 1:trainer.epochs
        for i in eachindex(X[:, 1])
            x = X[i, :]
            target = y[i]
            trainer.model = update!(trainer.model, x, target, η=trainer.η)
        end
    end

    return trainer.model
end
end
