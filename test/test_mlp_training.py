from micrograd.value import Value
from micrograd.mlp import Neuron, Layer, MLP
import random

def test_backward_pass():
  # Using a random seed as this toy dataset can run into problems 
  # depending on how the weights are initially randomly generated
  random.seed(5)


  mlp = MLP(3,[4,4,1])

  # Observed inputs 
  xs = [
    [2.0,3.0,-1.0], 
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
  ]
  # Target values
  ys = [1.0,-1.0,-1.0,1.0] 

  loss = 0
  for k in range(0,50):
    # Forward pass
    ypreds = [mlp(x) for x in xs]
    loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypreds))

    # Backward pass
    mlp.zero_grad()
    loss.backward()

    # Update params
    for p in mlp.parameters():
      p.data += -0.05 * p.grad

    print(k, "{:.6f}".format(loss.data))
  
  assert(loss.data < 0.05)

  # This example is close to first training example
  x_test = [2.2,3.15,-1.2]
  y_test = 1.0

  pred = mlp(x_test)
  assert(abs(y_test - pred.data) < 0.05)

if __name__=="__main__": 
  test_backward_pass()
