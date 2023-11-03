from micrograd.value import Value
from micrograd.mlp import Neuron, Layer, MLP


def test_neuron():
  n = Neuron(3)
  
  assert(isinstance(n.b, Value))
  assert(n.b.data > -1.000001 and n.b.data < 1.0 and n.b.data != 0.0)

  assert(len(n.w) == 3)

  for w in n.w:
    assert(isinstance(w, Value))
    assert(w.data > -1.000001 and w.data < 1.0 and w.data != 0.0)

  x = [2.0,1.0,0.0]
  out = n(x)

  wxb = (n.w[0] * x[0]) + (n.w[1] * x[1]) + (n.w[2] * x[2]) + n.b
  exp = wxb.relu()

  assert(out.data == exp.data)



def test_layer():
  # e.g. a hidden layer, reading from 3 inputs, and the hidden layer has 4 neurons
  l = Layer(3,4)
  
  assert(len(l.neurons) == 4)

  for n in l.neurons:
    assert(isinstance(n, Neuron))
    assert(len(n.w) == 3)

  inputs = [1.2,0.222,-0.34]
  outs = l(inputs)

  assert(len(outs)==4)

  assert(outs[0].data == l.neurons[0](inputs).data)
  assert(outs[1].data == l.neurons[1](inputs).data)
  assert(outs[2].data == l.neurons[2](inputs).data)
  assert(outs[3].data == l.neurons[3](inputs).data)

  # Test single output layer, out value should be returned unpacked (i.e. not in a one-element list)
  outL = Layer(4, 1)
  assert(len(outL.neurons) == 1)
  assert(isinstance(outL.neurons[0], Neuron))
  assert(len(outL.neurons[0].w) == 4)

  inputs = [0.33, 10.2, 0.0, -17.111]
  out = outL(inputs)

  assert(isinstance(out, Value))
  assert(out.data == outL.neurons[0](inputs).data)


def test_mlp():
  mlp = MLP(3,[4,4,1])

  assert(mlp.layer_sizes == [3,4,4,1])
  
  assert(isinstance(mlp.layers[0], Layer))
  assert(len(mlp.layers[0].neurons) == 4)

  assert(isinstance(mlp.layers[1], Layer))
  assert(len(mlp.layers[1].neurons) == 4)

  assert(isinstance(mlp.layers[2], Layer))
  assert(len(mlp.layers[2].neurons) == 1)

  x = [2.0, 3.0, -1.0]
  out = mlp(x)

  assert(isinstance(out, Value))

if __name__=="__main__": 
  test_neuron()
  test_layer()
  test_mlp()
