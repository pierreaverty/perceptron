include("../../src/model/Model.jl")
include("../../src/utils/Symbol.jl")
include("../../src/data/data.jl")
# --------------------
using Test
using .Model
using .Symbol
using .Data
# --------------------

@testset "Model" begin
    #=
    Test the Model module.
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

        # Test the type of the returned object and the size of the weights
        @test perceptron isa Perceptron
        @test length(perceptron.W) == 2
        @test perceptron.b isa Float64
        @test all(x -> x isa Float64, perceptron.W)
    end
    @testset "ɸ" begin
        #=
        Test the ɸ function.

        The function should return 1 if the input is greater than 0, otherwise 0.
        =#
        # Compute the dot product
        z = 0.59
        expected_result = z > 0 ? 1 : 0
        computed_result = ɸ(z)

        println("ɸ($z) = $computed_result")

        # Test the type of the returned object
        @test computed_result isa Int
    end
    @testset "P" begin
        #=
        Test the prediction function.

        The function should return the dot product of the weights and the input vector plus the bias.
        =#
        # Initialize a perceptron
        perceptron = init_perceptron(n=2)

        # Initialize a vector
        X = [1.0, 2.0]

        # Compute the dot product
        expected_result = ɸ(∑(perceptron.W, X) + perceptron.b)
        computed_result = P(perceptron.W, X, perceptron.b)
        println("P($(perceptron.W), $(X)) = $computed_result")

        # Test the type of the returned object and the value
        @test computed_result isa Int64
        @test computed_result == expected_result
    end
    @testset "update!" begin
        #=
        Test the update! function.

        The function should update the weights and bias of the perceptron.
        =#

        # Initialize a perceptron
        perceptron = init_perceptron(n=2)
        println("Initial weights: ", perceptron.W)
        println("Initial bias: ", perceptron.b)

        # # Initialize a vector
        X, Y = generate_data()

        # Update the perceptron
        update!(perceptron, X[1, :], Y[1], η=0.1)

        println("Updated weights: ", perceptron.W)
        println("Updated bias: ", perceptron.b)

        # Test the type of the returned object and the value
        @test all(x -> x isa Float64, perceptron.W)
        @test perceptron.b isa Float64
    end
end
