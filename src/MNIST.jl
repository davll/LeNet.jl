module MNIST

import MLDatasets
using PaddedViews: PaddedView
using FixedPointNumbers: N0f8

export train_dataset, test_dataset

function train_dataset(; padding=0, snorm=false)
    images, labels = MLDatasets.MNIST.traindata()
    images = preprocess_mnist_images(images; padding=padding, snorm=snorm)
    labels = preprocess_mnist_labels!(labels)
    scores = to_categorical(labels; snorm=snorm)
    images, labels, scores
end

function test_dataset(; padding=0, snorm=false)
    images, labels = MLDatasets.MNIST.testdata()
    images = preprocess_mnist_images(images; padding=padding, snorm=snorm)
    labels = preprocess_mnist_labels!(labels)
    scores = to_categorical(labels; snorm=snorm)
    images, labels, scores
end

function preprocess_mnist_images(images; padding=0, snorm=false)
    images = permutedims(images, (2,1,3))
    w, h, n = size(images)
    @assert w == 28
    @assert h == 28
    images = images .|> (snorm ? to_snorm : Float16)
    if padding > 0
        z = (snorm ? -1 : 0) |> Float16
        images = PaddedView(z, images, (28+2*padding,28+2*padding,n), (1+padding,1+padding,1))
        @assert size(images) == (28+2*padding, 28+2*padding, n)
    end
    images
end

function preprocess_mnist_labels!(labels)
    n = length(labels)
    labels[labels .== 0] .= 10
    labels
end

function to_categorical(labels; snorm=false)
    n = length(labels)
    scores = zeros(Float16, 10, n)
    for k = 1:10
        scores[k,:] = (labels .== k) .|> (snorm ? to_snorm : Float16)
    end
    scores
end

function to_snorm(x::N0f8)::Float16
    Float32(x) * 2.0 - 1.0
end

function to_snorm(x::Bool)::Float16
    x ? 1.0 : -1.0
end

end
