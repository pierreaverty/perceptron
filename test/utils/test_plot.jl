include("../../src/utils/Plot.jl")
include("../../src/data/Data.jl")
# --------------------
using Test
using .Plot
using .Data
# --------------------

@testset "Plot" begin
    #=
    Test the Plot module.
    =#
    @testset "plot_dataset" begin
        #=
        Test the plot_dataset function.

        The function should plot the data and the decision boundary.
        =#
        # Load the dataset
        X, Y = generate_data()

        # Plot the dataset
        plot_dataset(X, Y)

        @test isfile("res/dataset/dataset.png")
    end
end