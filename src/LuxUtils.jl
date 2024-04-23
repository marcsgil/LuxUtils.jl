module LuxUtils

using Lux, MLUtils
using Zygote, ADTypes, Optimisers
using Random, LinearAlgebra, Statistics
using ProgressMeter, JLD2, UnPack

export train

"""
TODO: Implement tensor board logging and early stopping.
"""

function save(train_state::Lux.Experimental.TrainState, smallest_loss, epoch, path)
    parameters = train_state.parameters |> cpu_device()
    states = train_state.states |> cpu_device()
    optimizer_state = train_state.optimizer_state |> cpu_device()
    step = train_state.step
    @save path parameters states optimizer_state step smallest_loss epoch
end

function train(model, x, y, epochs; rng=Random.default_rng(), optimiser=Adam(), device=gpu_device(), kwargs...)
    train_state = Lux.Experimental.TrainState(rng, model, optimiser; transform_variables=device)
    train(train_state::Lux.Experimental.TrainState, x, y, epochs; device, kwargs...)
end

function train(path::String, model, x, y, epochs; device=gpu_device(), kwargs...)
    file = jldopen(path)
    @unpack parameters, states, optimizer_state, step, smallest_loss, epoch = file
    close(file)
    train_state = Lux.Experimental.TrainState(model,
        parameters |> device,
        states |> device,
        optimizer_state |> device,
        step)
    start_epoch = epoch + 1
    train(train_state, x, y, epochs; device, smallest_loss, start_epoch, kwargs...)
end

function train(train_state::Lux.Experimental.TrainState, x, y, epochs;
    train_test_split=0.85,
    device=gpu_device(),
    batchsize=256,
    loss_function=(y, y_pred) -> mean(abs2.(y .- y_pred)),
    metrics=Tuple{Function,String}[],
    saving_dir="runs",
    smallest_loss=Inf,
    start_epoch=1,
    verbose=true,
    ad_backend=AutoZygote())

    push!(metrics, (loss_function, "Test loss"))

    @assert 0 < train_test_split ≤ 1

    mkpath(saving_dir)

    train_data, test_data = splitobs((x, y); at=train_test_split)
    train_loader = DataLoader(train_data; batchsize, shuffle=true)
    test_loader = DataLoader(test_data; batchsize)

    generate_showvalues(loss) = () -> [("Training loss", loss)]

    function _loss_function(model, ps, st, (x, y))
        y_pred, st = Lux.apply(model, x, ps, st)
        return loss_function(y, y_pred), st, (;)
    end

    mean_metrics = zeros(eltype(y), length(metrics))

    for epoch in start_epoch:start_epoch+epochs-1
        p = Progress(length(train_loader), showspeed=true, desc="Epoch: $epoch")
        train_loss = 0
        N = 0
        for (x, y) in train_loader
            x = x |> device
            y = y |> device
            gs, _train_loss, _, train_state = Lux.Experimental.compute_gradients(
                ad_backend, _loss_function, (x, y), train_state)
            train_state = Lux.Experimental.apply_gradients(train_state, gs)
            train_loss += _train_loss
            N += size(x, ndims(x))
            next!(p; showvalues=generate_showvalues(train_loss / N))
        end
        finish!(p)

        fill!(mean_metrics, 0)
        for (x, y) in test_loader
            x = x |> device
            y = y |> device
            y_pred, st = Lux.apply(train_state.model, x, train_state.parameters, train_state.states)
            for (n, metric) ∈ enumerate(metrics)
                mean_metrics[n] += metric[1](y, y_pred)
            end
        end
        mean_metrics /= length(test_loader)

        for (val, metric) ∈ zip(mean_metrics, metrics)
            printstyled("  $(metric[2]):  $val \n", color=:blue)
        end

        if mean_metrics[end] < smallest_loss
            if verbose
                @info "Test loss decreased: $smallest_loss -> $(mean_metrics[end]). Saving model..."
            end
            smallest_loss = mean_metrics[end]
            save(train_state, smallest_loss, epoch, joinpath(saving_dir, "best_model.jld2"))
        end

    end

    save(train_state, smallest_loss, start_epoch + epochs, joinpath(saving_dir, "last_model.jld2"))

    return train_state.parameters, train_state.states
end

end
