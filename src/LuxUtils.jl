module LuxUtils

using Reexport
@reexport using Lux, MLUtils, JLD2, Optimisers
using Zygote, Random, ProgressMeter

export train, save_model, compute_mean_metrics!

"""
    save_model(train_state, path)

Save the model's parameters and states to a file.

# Arguments
- `train_state`: A `TrainState` object containing the model's current parameters and states.
- `path`: A string specifying the file path where the model should be saved.

# Details
This function converts the model's parameters and states to CPU before saving them to the specified file using JLD2 format.
The saved file can be loaded later using the `jldopen` function:
```julia
ps, st = jldopen("model_checkpoint.jld2") do file
    file["parameters"], file["states"]
end
```

# Example
```julia
save_model(train_state, "model_checkpoint.jld2")
```
"""
function save_model(train_state, path)
    # Convert parameters and states to CPU before saving
    parameters = train_state.parameters |> cpu_device()
    states = train_state.states |> cpu_device()

    jldsave(path; parameters, states)
end

"""
    compute_mean_metrics!(dest, metrics, model, train_state, loader, device)

Compute mean metrics for the data in `loader` and store the results in `dest`.

# Arguments
- `dest`: Destination array to store the computed mean metrics.
- `metrics`: Array of tuples, each containing a metric function and its name.
- `model`: The Lux model being evaluated.
- `train_state`: A `TrainState` object containing the model's current parameters and states.
- `loader`: DataLoader for the test dataset.
- `device`: The device (CPU or GPU) on which to perform computations.

# Details
This function computes the mean of each specified metric over the entire test dataset.
The results are stored in-place in the `dest` array.

# Example
```julia
mean_metrics = zeros(Float32, length(metrics))
compute_mean_metrics!(mean_metrics, metrics, model, train_state, loader, cpu_device())
```
"""
function compute_mean_metrics!(dest, metrics, model, train_state, loader, device)
    ps = train_state.parameters
    st = Lux.testmode(train_state.states)  # Set model to test mode
    fill!(dest, 0)  # Initialize destination array
    for (x, y) in loader
        x = x |> device
        y = y |> device
        y_pred, _ = model(x, ps, st)  # Forward pass
        for (n, metric) ∈ enumerate(metrics)
            dest[n] += metric[1](y, y_pred)  # Accumulate metric values
        end
    end
    dest ./= length(loader)  # Compute mean
end

"""
    train(x, y, epochs, model[, ps, st]; kwargs...)

Train a Lux model with the given data and parameters.

# Arguments
- `x`: Input data.
- `y`: Target data.
- `epochs`: Number of training epochs.
- `model`: The Lux model to train.
- `ps`: Initial parameters of the model.
- `st`: Initial states of the model.

If `ps` and `st` are not provided, the model will be initialized using `ps, st = Lux.setup(rng, model) |> device`.

# Keyword Arguments
- `train_test_split=0.85`: Proportion of data to use for training.
- `device=cpu_device()`: Device to use for computations (CPU or GPU).
- `batchsize=256`: Batch size for training.
- `opt=Adam()`: Optimizer to use for training. Other optimizers can be used from the `Optimisers` package (reexported).
- `loss_function=MSELoss()`: Loss function for training.
- `metrics::Vector{Tuple{Function,String}}=[]`: Additional metrics to compute during training.
- `model_saving_path=""`: If provided, path to save model checkpoints. Only the best model is saved.
        The results can be latter loaded using the JLD2 package (reexported):

```julia
    ps, st = jldopen(path2model_checkpoint) do file
        file["parameters"], file["states"]
    end
```

- `logging_path=""`: If provided, path to save training logs (CSV format).
- `ad_backend=AutoZygote()`: Automatic differentiation backend to use.
- `patience=Inf`: Number of epochs with no substantial improvement after which training will be stopped.
- `rtol=zero(float(eltype(y)))`: Relative tolerance for improvement in loss for early stopping.
- `atol=zero(float(eltype(y)))`: Absolute tolerance for improvement in loss for early stopping.
- `verbose=true`: Whether to print training progress (Loss, metrics, epoch, etc.).
- `rng=Xoshiro(0)`: Random number generator for parameter initialization and data shuffling.

# Returns
A tuple containing the trained model's parameters and states.

# Examples
```julia
x = randn(Float32, 10, 512)
y = sum(x, dims=1) .+ 0.1f0 * randn(Float32, 1, 512)

model = Chain(
    Dense(10, 5, relu),
    Dense(5, 1)
)

epochs = 5

# Train the model with default initialization
ps, st = train(x, y, epochs, model, batchsize=4);

# Train the model with pre-initialized parameters
# This method is useful for resuming training
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = train(x, y, epochs, model, ps, st, batchsize=4);
```
"""
function train(x, y, epochs, model, ps, st;
    train_test_split=0.85,
    device=cpu_device(),
    batchsize=256,
    opt=Adam(),
    loss_function=MSELoss(),
    metrics=Tuple{Function,String}[],
    model_saving_path="",
    logging_path="",
    ad_backend=AutoZygote(),
    patience=Inf,
    rtol=zero(float(eltype(y))),
    atol=zero(float(eltype(y))),
    verbose=true,
    rng=Xoshiro(0))

    train_state = Training.TrainState(model, ps, st, opt)
    loss = [(loss_function, "Test loss")]
    metrics_and_loss = vcat(loss, metrics)  # Add loss function to metrics

    @assert 0 < train_test_split ≤ 1

    # Split data into train and test sets
    train_data, test_data = splitobs(shuffleobs(rng, (x, y)); at=train_test_split)
    train_loader = DataLoader(train_data; batchsize, shuffle=true, rng)
    test_loader = DataLoader(test_data; batchsize)

    generate_showvalues(loss) = () -> [("Training loss", loss)]

    mean_metrics = zeros(float(eltype(y)), length(metrics_and_loss))
    # Compute initial test loss
    compute_mean_metrics!((@view mean_metrics[begin:begin]), loss, model,
        train_state, test_loader, device)

    smallest_loss = mean_metrics[begin]

    # Create directory for model saving if path is provided
    if !isempty(model_saving_path)
        mkpath(dirname(model_saving_path))
    end

    # Set up logging if path is provided
    if !isempty(logging_path)
        mkpath(dirname(logging_path))

        if isfile(logging_path)
            error("Logging path already exists. Please provide a new path.")
        end

        open(logging_path, "a") do file
            write(file, "Epoch, " * join(last.(metrics_and_loss), ", ") * "\n")
        end
    end

    early_stopping_counter = 0
    best_epoch = 0

    for epoch in 1:epochs

        if verbose
            p = Progress(length(train_loader), showspeed=true, desc="Epoch: $epoch")
        end

        train_loss = zero(float(eltype(y)))
        for (N, (x, y)) in enumerate(train_loader)
            x = x |> device
            y = y |> device
            # Perform a single training step
            grads, _train_loss, stats, train_state = Training.single_train_step!(
                ad_backend, loss_function, (x, y), train_state)
            train_loss += _train_loss

            if verbose
                next!(p; showvalues=generate_showvalues(train_loss / N))
            end
        end
        if verbose
            finish!(p)
        end

        # Compute metrics on test set
        compute_mean_metrics!(mean_metrics, metrics_and_loss, model, train_state, test_loader, device)

        if verbose
            # Print metrics
            for (val, metric) ∈ zip(mean_metrics, metrics_and_loss)
                printstyled("  $(metric[2]):  $val \n", color=:blue)
            end
        end

        # Log metrics if logging is enabled
        if !isempty(logging_path)
            open(logging_path, "a") do file
                write(file, "$epoch, " * join(mean_metrics, ", ") * "\n")
            end
        end

        δ = smallest_loss - mean_metrics[begin]  # Improvement in loss

        if δ > 0  # If there's an improvement
            if verbose
                @info "Test loss decreased: $smallest_loss -> $(mean_metrics[begin])"
            end

            smallest_loss = mean_metrics[begin]
            best_epoch = epoch

            # Save model if path is provided
            if !isempty(model_saving_path)
                save_model(train_state, model_saving_path)
            end

            # Check if improvement is significant
            if δ ≥ max(atol, rtol * smallest_loss)
                early_stopping_counter = 0
            else
                early_stopping_counter += 1
            end
        else
            early_stopping_counter += 1
        end

        # Check for early stopping
        if early_stopping_counter ≥ patience
            if verbose
                @info "Early stopping. Best epoch: $best_epoch"
            end
            break
        end

    end

    return train_state.parameters, train_state.states
end

# Convenience method for training with default initialization
function train(x, y, epochs, model; device=cpu_device(), rng=Xoshiro(0), kwargs...)
    ps, st = Lux.setup(rng, model) |> device
    train(x, y, epochs, model, ps, st; device, rng, kwargs...)
end

end