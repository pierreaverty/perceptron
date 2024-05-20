# --------------------
using .Model
using .Trainer
using .Data
# --------------------
@testset "Trainer" begin
    #=
    Test the Trainer module.
    =#
    @testset "PerceptronTrainer" begin
        #=
        Test the PerceptronTrainer struct.

        The struct should contain the model, dataset, epochs, and learning rate.
        =#
        # Initialize the model
        model = init_perceptron(n=2)

        # Initialize the dataset
        dataset = generate_data()

        # Initialize the trainer
        trainer = init_perceptron_trainer(
            model,
            dataset
        )

        # Test the type of the returned object
        @test trainer isa PerceptronTrainer
        @test trainer.model isa Perceptron
        @test trainer.dataset isa Tuple
        @test trainer.epochs isa Int64
        @test trainer.η isa Float64
    end
    @testset "train!" begin
        #=
        Test the train! function.

        The function should update the model weights and bias.
        =#
        # Initialize the model
        model = init_perceptron(n=2)

        # Initialize the dataset
        dataset = generate_data()

        # Initialize the trainer
        trainer = init_perceptron_trainer(
            model,
            dataset,
            epochs=100
        )

        println("Initial weights: ", trainer.model.W)
        # Train the model
        trained_model = train!(trainer)
        println("Trained weights: ", trained_model.W)

        # Test the type of the returned object
        @test P(trained_model.W, [1.0, 1.0], trained_model.b) == 0
        @test P(trained_model.W, [3.0, 3.0], trained_model.b) == 1
        @test trained_model isa Perceptron
        @test trained_model.W isa Array{Float64,1}
        @test trained_model.b isa Float64
    end
end
