using Perceptron
using LinearAlgebra
using Test

@testset "Neuron Initialization" begin
    neuron = Neuron(randn(2, 1), 0.0, x -> x)

    @testset "Weights" begin
        @test size(neuron.W) == (2, 1)
    end

    @testset "Bias" begin
        @test neuron.β == 0.0
    end

    @testset "Activation function" begin
        @test neuron.g(1) == 1
    end
end

@testset "Neuron Forward Pass" begin
    neuron = Neuron(randn(2, 1), 0.0, x -> x)

    @testset "Input" begin
        @test y(neuron, [1.0, 2.0]) == dot(neuron.W, [1.0, 2.0]) + neuron.β
    end
end

@testset "Activation Function" begin
    @testset "Heaviside Step Function" begin
        @test Activation.θ(0.0) == 1
        @test Activation.θ(-0.1) == 0
    end
end
