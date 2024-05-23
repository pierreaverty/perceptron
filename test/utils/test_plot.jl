# --------------------
using .Plot
using .Data
using .Model
using .Trainer
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

        @test isfile("./val/dataset/dataset.png")
    end
    @testset "plot_decision_boundary" begin
        #=
        Test the plot_decision_boundary function.

        The function should plot the decision boundary.
        =#
        # Load the dataset
        dataset = generate_data()

        # Initialize a perceptron and the trainer
        perceptron = init_perceptron(n=2)
        trainer = init_perceptron_trainer(
            perceptron,
            dataset,
            epochs=100
        )

        # Train the model
        train!(trainer)

        # Split the dataset
        X, Y = dataset

        # Plot the decision boundary
        plot_decision_boundary(perceptron.W, perceptron.b, X, Y)

        @test isfile("./val/model/decision_boundary.png")
    end
end
