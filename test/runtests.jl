using Perceptron
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
