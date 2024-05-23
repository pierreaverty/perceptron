# --------------------
using .Model
using .Activation
using .Math
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
        println("Testing Perceptron struct initilisation...")
        println("Perceptron of size 2 successfully initialised:")
        println("W = $(perceptron.W)")
        println("b = $(perceptron.b)")

        # Test the type of the returned object and the size of the weights
        @test perceptron isa Perceptron
        @test length(perceptron.W) == 2
        @test perceptron.b isa Float64
        @test all(x -> x isa Float64, perceptron.W)
    end
    @testset "predict" begin
        #=
        Test the prediction function.

        The function should return the dot product of the weights and the input vector plus the bias.
        =#
        # Initialize a perceptron
        perceptron = init_perceptron(n=2)

        # Initialize a vector
        X = [1.0, 2.0]

        # Compute the dot product
        expected_result = H(∑(perceptron.W, X) + perceptron.b)
        computed_result = predict(perceptron, X)
        println("Testing predict function...")
        println("predict(X) = $computed_result")

        # Test the type of the returned object and the value
        @test computed_result isa Int64
        @test computed_result == expected_result
    end
end
