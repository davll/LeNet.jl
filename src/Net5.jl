module Net5

using Knet, JLD2, FileIO
using Statistics: mean
import ..load_model, ..save_model, ..loss, ..fit

struct LeNet5
    conv1_w
    conv1_b
    conv2_w
    conv2_b
    conv3_w
    conv3_b
    fc4_w
    fc4_b
    fc5_w
    fc5_b
end

function LeNet5()
    LeNet5(
        param(5,5,1,6), param0(1,1,6,1),
        param(5,5,6,16), param0(1,1,16,1),
        param(5,5,16,120), param0(1,1,120,1),
        param(84,120), param0(84),
        param(10,84), param0(10),
    )
end

function (nn::LeNet5)(x; f=tanh)
    # check input
    if ndims(x) == 3
        w, h, n = size(x)
        x = reshape(x, (w, h, 1, n))
    end
    @assert ndims(x) == 4
    w, h, d, n = size(x)
    @assert w == 32
    @assert h == 32
    @assert d == 1
    # forward pass
    # conv1: 32x32x1 -> 28x28x6 -> 14x14x6
    x = conv4(nn.conv1_w, x; padding=0, stride=1, mode=0) .+ nn.conv1_b
    @assert size(x) == (28,28,6,n)
    x = f.(x)
    # average pool
    x = pool(x; window=2, padding=0, stride=2, mode=1)
    @assert size(x) == (14,14,6,n)
    # conv2: 14x14x6 -> 10x10x16 -> 5x5x16
    x = conv4(nn.conv2_w, x; padding=0, stride=1, mode=0) .+ nn.conv2_b
    @assert size(x) == (10,10,16,n)
    x = f.(x)
    # average pool
    x = pool(x; window=2, padding=0, stride=2, mode=1)
    @assert size(x) == (5,5,16,n)
    # conv3: 5x5x16 -> 1x1x120
    x = conv4(nn.conv3_w, x; padding=0, stride=1, mode=0) .+ nn.conv3_b
    @assert size(x) == (1,1,120,n)
    x = f.(x)
    # reshape: 1x1x120 -> 120
    x = reshape(x,:,n)
    @assert size(x) == (120,n)
    # fc4: 120 -> 84
    x = nn.fc4_w * x .+ nn.fc4_b
    @assert size(x) == (84,n)
    x = f.(x)
    # fc5: 84 -> 10
    x = nn.fc5_w * x .+ nn.fc5_b
    @assert size(x) == (10,n)
    # final
    softmax(x)
end

function loss(nn::LeNet5, x, y)
    #mean(sum((nn(x) .- y) .^ 2, dims=1))
    x = nn(x)
    e = eps(eltype(x))
    (-y .* log.(x .+ e) .- (1 .- y) .* log.(1 .- x .+ e)) |> a->sum(a,dims=1) |> mean
end

(nn::LeNet5)(x, y) = loss(nn, x, y)

function fit(nn::LeNet5, data)
    sgd(nn, data; lr=0.01, params=params(nn))
end

function save_model(path, model::LeNet5)
    if Knet.gpu() >= 0
        cpu = cpucopy
    else
        cpu = identity
    end
    jldopen(path, "w") do file
        file["lenet5/conv1_w"] = model.conv1_w |> value |> cpu
        file["lenet5/conv1_b"] = model.conv1_b |> value |> cpu
        file["lenet5/conv2_w"] = model.conv2_w |> value |> cpu
        file["lenet5/conv2_b"] = model.conv2_b |> value |> cpu
        file["lenet5/conv3_w"] = model.conv3_w |> value |> cpu
        file["lenet5/conv3_b"] = model.conv3_b |> value |> cpu
        file["lenet5/fc4_w"]   = model.fc4_w   |> value |> cpu
        file["lenet5/fc4_b"]   = model.fc4_b   |> value |> cpu
        file["lenet5/fc5_w"]   = model.fc5_w   |> value |> cpu
        file["lenet5/fc5_b"]   = model.fc5_b   |> value |> cpu
    end;
end

function load_model(::Type{LeNet5}, path)::LeNet5
    if Knet.gpu() >= 0
        gpu = gpucopy
    else
        gpu = identity
    end
    jldopen(path, "r") do file
        LeNet5(
            file["lenet5/conv1_w"] |> gpu,
            file["lenet5/conv1_b"] |> gpu,
            file["lenet5/conv2_w"] |> gpu,
            file["lenet5/conv2_b"] |> gpu,
            file["lenet5/conv3_w"] |> gpu,
            file["lenet5/conv3_b"] |> gpu,
            file["lenet5/fc4_w"]   |> gpu,
            file["lenet5/fc4_b"]   |> gpu,
            file["lenet5/fc5_w"]   |> gpu,
            file["lenet5/fc5_b"]   |> gpu,
        )
    end
end

end
