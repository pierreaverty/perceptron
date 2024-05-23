# --------------------
using .Activation
# --------------------

@testset "Activation" begin
    #=
    Test the Activation module.
    =#
    @testset "H" begin
        #=
        Test the heaviside step function.

        The function should return 1 if the input is greater than 0, otherwise 0.
        =#
        # Compute the dot product
        z = 0.59
        expected_result = z > 0 ? 1 : 0
        computed_result = H(z)
        println("Testing heaviside step function...")
        println("H($z) = $computed_result")

        # Test the type of the returned object
        @test computed_result isa Int
    end
end
