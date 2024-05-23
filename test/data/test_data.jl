# --------------------
using .Data
using .Model
# --------------------
@testset "Data" begin
    #=
    Test the Data module.
    =#
    @testset "generate_data" begin
        #=
        Test the generate_data function.

        The function should return the data and labels.
        =#
        # Load the data
        X, Y = generate_data()
        # Print the data and labels
        println("Testing data generation function...")
        println("Data:")
        println(X)
        println("Labels:")
        println(Y)

        # Test the type of the returned objects
        @test X isa Matrix{Float64}
        @test Y isa Vector{Int64}
        @test size(X) == (100, 2)
        @test length(Y) == 100
        @test all(x -> x isa Float64, X)
        @test all(x -> x isa Int64, Y)
    end
end
