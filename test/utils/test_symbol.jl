include("../../src/utils/Symbol.jl")

# --------------------
using Test
using .Symbol
# --------------------

@testset "Symbol" begin
    #=
    Test the Symbol module.
    =#
    @testset "∑" begin
        #=
        Test the ∑ function.

        The function should return the dot product of two vectors.
        =#
        # Initialize two vectors
        X = [1.0, 2.0, 3.0]
        Y = [4.0, 5.0, 6.0]

        # Compute the dot product
        dot_product = ∑(X, Y)

        # Print the dot product
        println("∑($(X),$(Y) = $dot_product")

        # Test the type of the returned object
        @test dot_product isa Float64
        @test dot_product == 32.0
    end
end
