# --------------------
using .Optimiser
using .Model
using .Data
# --------------------

@testset "Optimiser" begin
    #=
    Test the Optimiser module.
    =#
    @testset "update!" begin
        #=
        Test the update! function.

        The function should update the weights and bias of the perceptron.
        =#

        # Initialize a perceptron
        perceptron = init_perceptron(n=2)
        println("Testing update function...")
        println("Initial weights: ", perceptron.W)
        println("Initial bias: ", perceptron.b)

        # # Initialize a vector
        X, Y = generate_data()

        # Update the perceptron
        perceptron_update!(perceptron, X[1, :], Y[1], η=0.1)

        println("Updated weights: ", perceptron.W)
        println("Updated bias: ", perceptron.b)

        # Test the type of the returned object and the value
        @test all(x -> x isa Float64, perceptron.W)
        @test perceptron.b isa Float64
    end
end
