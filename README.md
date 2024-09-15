# LuxUtils

LuxUtils is a Julia module that simplifies the process of training neural networks using the [`Lux`](https://github.com/LuxDL/Lux.jl) framework. It provides a high-level interface for training models, handling common tasks such as data splitting, progress tracking, early stopping, logging, and model saving.

## Features

- Easy-to-use training function with sensible defaults
- Training progress visualization with a progress bar for each epoch and a list of metrics in the REPL
- Automatic train/test split
- Progress tracking with customizable metrics
- Early stopping
- Model checkpointing
- Logging of training progress
- Reexports [`Lux`](https://github.com/LuxDL/Lux.jl), [`MLUtils`](https://github.com/JuliaML/MLUtils.jl), [`JLD2`](https://github.com/JuliaIO/JLD2.jl), [`Optimisers`](https://github.com/FluxML/Optimisers.jl)

## Installation

LuxUtils is not yet registered as an official Julia package. To install it, you can add it directly from this GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/marcsgil/LuxUtils.jl.git")
```

## Usage

Here's a basic example of how to use LuxUtils to train a simple model:

```julia
using LuxUtils
using Random

# Generate some dummy data
x = randn(Float32, 10, 1000)
y = sum(x, dims=1) .+ 0.1f0 * randn(Float32, 1, 1000)

# Define the model
model = Chain(
    Dense(10, 5, relu),
    Dense(5, 1)
)

# Train the model
ps, st = train(x, y, 10, model, batchsize=32)
```

### MNIST Example

Here's an example of how to use LuxUtils to train a model on the MNIST dataset:

```julia
# Load the required packages
# Be sure to also install the MLDatasets and OneHotArrays packages
using LuxUtils, MLDatasets, OneHotArrays

# Loading the MNIST dataset
dataset = MNIST()
x = reshape(dataset.features, 784, :)
y = onehotbatch(dataset.targets, 0:9)

# Defining the model
model = Chain(
    Dense(784, 15, relu),
    Dense(15, 10),
    softmax
)

# Defining the loss function
loss_function = CrossEntropyLoss(; logits=Val(true))

# Defining the accuracy function
function accuracy(y, y_hat)
    y = onecold(y)
    y_hat = onecold(y_hat)
    mean(y .== y_hat)
end

# Defining the metrics
metrics = [(accuracy, "Accuracy")]

epochs = 10

# Training the model
ps, st = train(x, y, epochs, model; loss_function, metrics);
```

### Advanced Usage

LuxUtils provides many options to customize the training process:

```julia
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
```

## API Reference

### train

```julia
train(x, y, epochs, model[, ps, st]; kwargs...)
```

Train a Lux model with the given data and parameters.

#### Arguments

- `x`: Input data
- `y`: Target data
- `epochs`: Number of training epochs
- `model`: The Lux model to train
- `ps`: (Optional) Initial parameters of the model
- `st`: (Optional) Initial states of the model

#### Keyword Arguments

- `train_test_split=0.85`: Proportion of data to use for training
- `device=cpu_device()`: Device to use for computations (CPU or GPU)
- `batchsize=256`: Batch size for training
- `opt=Adam()`: Optimizer to use for training
- `loss_function=MSELoss()`: Loss function for training
- `metrics=[]`: Additional metrics to compute during training
- `model_saving_path=""`: Path to save model checkpoints
- `logging_path=""`: Path to save training logs
- `ad_backend=AutoZygote()`: Automatic differentiation backend to use
- `patience=Inf`: Number of epochs with no improvement after which training will be stopped
- `rtol=0.0`: Relative tolerance for improvement in loss for early stopping
- `atol=0.0`: Absolute tolerance for improvement in loss for early stopping
- `verbose=true`: Whether to print training progress
- `rng=Xoshiro(0)`: Random number generator for initialization and shuffling

#### Returns

A tuple containing the trained model's parameters and states.

### save_model

```julia
save_model(train_state, path)
```

Save the model's parameters and states to a file.

### compute_mean_metrics!

```julia
compute_mean_metrics!(dest, metrics, model, train_state, loader, device)
```

Compute mean metrics for the data in `loader` and store the results in `dest`.

## Contributing

Contributions to LuxUtils are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
