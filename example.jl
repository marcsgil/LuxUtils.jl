using LuxUtils, MLDatasets, OneHotArrays

#Loading the MNIST dataset
dataset = MNIST()
x = reshape(dataset.features, 784, :)
y = onehotbatch(dataset.targets, 0:9)

#Defining the model
model = Chain(
    Dense(784, 15, relu),
    Dense(15, 10),
    softmax
)

#Defining the loss function
loss_function = CrossEntropyLoss(; logits=Val(true))

#Defining the accuracy function
function accuracy(y, y_hat)
    y = onecold(y)
    y_hat = onecold(y_hat)
    mean(y .== y_hat)
end

#Defining the metrics
metrics = [(accuracy, "Accuracy")]

#Training the model
ps, st = train(x, y, 10, model; loss_function, metrics);

