include("../../src/model/Model.jl")

# --------------------
using Test
using ..Model
# --------------------

@testset "Perceptron" begin
    #=
    Test the Perceptron struct.

    The function should return a Perceptron struct with n weights and a bias.
    =#
    @testset "init_perceptron" begin
        #=
        Test the init_perceptron function with n=2.

        The function should return a Perceptron object with 2 weights and a bias.
        =#
        # Initialize a perceptron
        perceptron = init_perceptron(n=2)

        # Print the initialized perceptron
        println("Perceptron of size 2 successfully initialized:")
        println("W = $(perceptron.W)")
        println("b = $(perceptron.b)")

        # Test the type of the returned object
        @test perceptron isa Perceptron
        @test length(perceptron.W) == 2
        @test perceptron.b isa Float64
        @test all(x -> x isa Float64, perceptron.W)
    end
end
