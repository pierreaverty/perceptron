using Perceptron
using LinearAlgebra
using Test

@testset "Neuron Initialization" begin
    neuron = Neuron((2, 1), Activation.θ)

    @testset "Weights" begin
        @test size(neuron.W) == (1, 2)
    end

    @testset "Bias" begin
        @test neuron.β == [1.0;;]
    end

    @testset "Activation function" begin
        @test neuron.g(1.0) == 1
    end
end

@testset "Neuron Forward Pass" begin
    neuron = Neuron((2, 1), x -> x)

    @testset "Single Output" begin
        x = [1.0 2.0]

        @test y(neuron, x) == neuron.W * x' .+ neuron.β
    end

    @testset "Multiple Outputs" begin
        W = zeros(2, 2)
        β = [1.0 1.0]
        neuron = Neuron((2, 2), x -> x)

        x = [1.0 2.0]
        z = neuron.W * x' .+ neuron.β

        @test y(neuron, x) == z
    end
end

@testset "Activation Function" begin
    @testset "Heaviside Step Function" begin
        @test Activation.θ(0.0) == 1
        @test Activation.θ(-0.1) == 0
    end
end

@testset "Optimization" begin
    @testset "Perceptron Algorithm Update Rule" begin
        neuron = Neuron((2, 1), x -> x)
        opt = Optimization.PerceptronOptimizer(0.1)

        W = copy(neuron.W)
        β = copy(neuron.β)

        x = [1.0 2.0]
        z = [1.0;;]

        z̄ = y(neuron, x)

        ΔW, Δβ = Optimization.update(opt, neuron.W, neuron.β, (x, z), z̄)

        @test ΔW == W .+ 0.1 .* (z - z̄) .* x
        @test Δβ == β .+ 0.1 .* (z - z̄)
    end
end
