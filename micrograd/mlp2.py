from micrograd.value import Value
import random

class Neuron():
  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(1,-1)) for n in range(nin)]
    self.b = Value(random.uniform(1,-1))
    self.nonlin = nonlin
    
  def __call__(self, x):
    # compute w*x + b
    pairwise = zip(self.w,x)
    val = self.b + sum([wi * xi for wi, xi in pairwise])
    # apply activation fn
    out = val.relu() if self.nonlin else val
    return out

  def parameters(self):
    return self.w + [self.b]

class Layer():
  def __init__(self, nin, nout, nonlin):
    self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

  def __call__(self, x):
    # return the list of outputs/activations for all neurons in this layer

    outs = [n(x) for n in self.neurons]
    # If layer has a single output (e.g. output layer), unpack this out of the list
    return outs[0] if len(self.neurons) == 1 else outs

  def parameters(self):
    params = []
    for n in self.neurons:
      params.extend(n.parameters())

    return params

class MLP():
  def __init__(self, nin, layers):
    self.layer_sizes = [nin] + layers # layers is a list, e.g. [3] + [4,4,1] = [3,4,4,1]
    # the input layer is not stored as a layer, inputs are passed to each layer in the __call__
    # we don't want to apply nonlinear activation function to the output layer
    self.layers = [Layer(self.layer_sizes[i], self.layer_sizes[i+1], nonlin=i!=len(layers)-1) for i in range(len(layers))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    params = []
    for l in self.layers:
      params.extend(l.parameters())

    return params

  def zero_grad(self):
    for p in self.parameters():
      # Zero all the gradients before each backward pass
      p.grad = 0.0
