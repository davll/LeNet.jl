module Net5

using Knet, JLD2, FileIO
using Statistics: mean
import ..load_model, ..save_model, ..loss, ..fit

struct LeNet5
    conv1_w; conv1_b
    conv2_w; conv2_b
    conv3_w; conv3_b
    fc4_w; fc4_b
    fc5_w; fc5_b
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

struct LeNet5Iterator
    nn
    x0
end

function (nn::LeNet5)(x)
    # check input
    if ndims(x) == 2
        w, h = size(x)
        x = reshape(x, (w, h, 1, 1))
    elseif ndims(x) == 3
        w, h, n = size(x)
        x = reshape(x, (w, h, 1, n))
    end
    @assert ndims(x) == 4
    LeNet5Iterator(nn, x)
end

function Base.iterate(it::LeNet5Iterator)
    Base.iterate(it, (it.x0, :conv1))
end

function Base.iterate(it::LeNet5Iterator, state)
    x, layer = state
    nn = it.nn
    if layer == :conv1
        # convolution with 5x5 kernel, channel 1 -> 6
        x = conv4(nn.conv1_w, x; padding=0, stride=1, mode=0) .+ nn.conv1_b
        x = tanh.(x)
        return ((:conv1, x), (x, :pool1))
    elseif layer == :pool1
        # average pooling
        x = pool(x; window=2, padding=0, stride=2, mode=1)
        return ((:pool1, x), (x, :conv2))
    elseif layer == :conv2
        # convolution with 5x5 kernel, channel 6 -> 16
        x = conv4(nn.conv2_w, x; padding=0, stride=1, mode=0) .+ nn.conv2_b
        x = tanh.(x)
        return ((:conv2, x), (x, :pool2))
    elseif layer == :pool2
        # average pooling
        x = pool(x; window=2, padding=0, stride=2, mode=1)
        return ((:pool2, x), (x, :conv3))
    elseif layer == :conv3
        # convolution with 5x5 kernel, channel 16 -> 120
        x = conv4(nn.conv3_w, x; padding=0, stride=1, mode=0) .+ nn.conv3_b
        x = tanh.(x)
        return ((:conv3, x), (x, :fc4))
    elseif layer == :fc4
        x = reshape(x,:,size(x,4))
        x = nn.fc4_w * x .+ nn.fc4_b
        x = tanh.(x)
        return ((:fc4, x), (x, :fc5))
    elseif layer == :fc5
        x = nn.fc5_w * x .+ nn.fc5_b
        return ((:fc5, x), (x, :prob))
    elseif layer == :prob
        x = softmax(x)
        return ((:prob, x), nothing)
    else
        throw(Exception("Invalid Layer"))
    end
end

Base.iterate(it::LeNet5Iterator, ::Nothing) = nothing
Base.IteratorSize(it::LeNet5Iterator) = Base.HasLength()
Base.length(it::LeNet5Iterator) = 8
Base.last(it::LeNet5Iterator) = Iterators.drop(it,7) |> collect |> last |> last

function loss(nn::LeNet5, x, y)
    #mean(sum((nn(x) .- y) .^ 2, dims=1))
    x = last(nn(x))
    e = eps(eltype(x))
    (-y .* log.(x .+ e) .- (1 .- y) .* log.(1 .- x .+ e)) |> a->sum(a,dims=1) |> mean
end

function fit(nn::LeNet5, data)
    sgd((x,y) -> loss(nn,x,y), data; lr=0.01, params=params(nn))
end

function save_model(path, model::LeNet5)
    arr(a) = convert(Array{Float32}, a)
    jldopen(path, "w") do file
        file["lenet5/conv1_w"] = model.conv1_w |> value |> arr
        file["lenet5/conv1_b"] = model.conv1_b |> value |> arr
        file["lenet5/conv2_w"] = model.conv2_w |> value |> arr
        file["lenet5/conv2_b"] = model.conv2_b |> value |> arr
        file["lenet5/conv3_w"] = model.conv3_w |> value |> arr
        file["lenet5/conv3_b"] = model.conv3_b |> value |> arr
        file["lenet5/fc4_w"]   = model.fc4_w   |> value |> arr
        file["lenet5/fc4_b"]   = model.fc4_b   |> value |> arr
        file["lenet5/fc5_w"]   = model.fc5_w   |> value |> arr
        file["lenet5/fc5_b"]   = model.fc5_b   |> value |> arr
    end;
end

function load_model(::Type{LeNet5}, path; atype=nothing)::LeNet5
    if atype === nothing
        atype = (Knet.gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
    end
    arr(a) = convert(atype, a)
    jldopen(path, "r") do file
        LeNet5(
            file["lenet5/conv1_w"] |> arr,
            file["lenet5/conv1_b"] |> arr,
            file["lenet5/conv2_w"] |> arr,
            file["lenet5/conv2_b"] |> arr,
            file["lenet5/conv3_w"] |> arr,
            file["lenet5/conv3_b"] |> arr,
            file["lenet5/fc4_w"]   |> arr,
            file["lenet5/fc4_b"]   |> arr,
            file["lenet5/fc5_w"]   |> arr,
            file["lenet5/fc5_b"]   |> arr,
        )
    end
end

end
