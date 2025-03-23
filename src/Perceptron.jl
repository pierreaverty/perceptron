module Perceptron

using LinearAlgebra

export Neuron,
    Activation,
    Optimization,
    y,
    fit!

include("Activation.jl")
include("Optimization.jl")
include("model.jl")

end # module Perceptron
