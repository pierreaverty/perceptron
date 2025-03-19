module Perceptron

using LinearAlgebra

export Neuron,  Activation, y

include("activation.jl")
include("model.jl")

end # module Perceptron
