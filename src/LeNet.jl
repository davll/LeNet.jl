"""
LeCun's Neural Network

Publication:
    'Gradient-based learning applied to document recognition'
    Yann LeCun et al.
    1998
    http://yann.lecun.com/exdb/lenet/

References:
    https://cs231n.github.io/convolutional-networks/
    https://www.d2l.ai/chapter_convolutional-neural-networks/lenet.html

Notes:
    Why the LeNet-5 uses 32x32 image as input instead of 28x28?
    https://stackoverflow.com/questions/28525436/why-the-lenet5-uses-32%C3%9732-image-as-input
"""
module LeNet

export LeNet5
export save_model, load_model, loss, fit

function save_model end
function load_model end
function loss end
function fit end

include("MNIST.jl")
include("Net5.jl")

import .Net5: LeNet5

end
