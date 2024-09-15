using LuxUtils
using Test, Random, Statistics

n_samples = 100
n_features = 10

x = randn(Float32, n_features, n_samples)
y = sum(x, dims=1) .+ 0.1f0 * randn(Float32, 1, n_samples)

model = Chain(
    Dense(10 => 5, relu),
    Dense(5 => 1)
)

verbose = false
batchsize = 4

# Test the main train function
@testset "Train Function" begin
    # Test with default initialization
    ps1, st1 = train(x, y, 5, model; batchsize, verbose)

    rng = Xoshiro(0)
    # Test with pre-initialized parameters
    ps2, st2 = Lux.setup(rng, model)
    ps2, st2 = train(x, y, 5, model, ps2, st2; batchsize, verbose, rng)

    @test ps1 == ps2
    @test st1 == st2

    # Test early stopping
    ps, st = train(x, y, 10, model; batchsize, verbose, patience=2)
    @test ps isa NamedTuple
    @test st isa NamedTuple
end

# Test model saving and loading
@testset "Model Saving and Loading" begin
    mktempdir() do dir
        model_saving_path = joinpath(dir, "model_checkpoint.jld2")

        # Train and save model
        ps, st = train(x, y, 5, model; batchsize, verbose, model_saving_path)

        # Test if file is created
        @test isfile(model_saving_path)
    end
end

# Test metrics computation
@testset "Logging" begin
    mae(ŷ, y) = Statistics.mean(abs, ŷ .- y)

    metrics = [(mae, "MAE")]

    mktempdir() do dir
        logging_path = joinpath(dir, "log.csv")

        # Train and log metrics
        ps, st = train(x, y, 5, model; batchsize, verbose, logging_path, metrics)

        # Test if file is created
        @test isfile(logging_path)

        # Test if metrics are logged
        @test first(eachline(logging_path)) == "Epoch, Test loss, MAE"
    end
end