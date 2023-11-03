from micrograd.value import Value

def test_backward_add():
  v1 = Value(2.0)
  v2 = Value(3.0)

  v3 = v1 + v2
  v3.grad = -5.0

  assert(v1.grad == 0.0)
  assert(v2.grad == 0.0)
  v3._backward()  

  # Gradient flows back unchanged to each child node of addition operation
  assert(v1.grad == -5.0)
  assert(v2.grad == -5.0)

def test_backward_mul():
  v1 = Value(2.0)
  v2 = Value(3.0)

  v3 = v1 * v2
  v3.grad = 1.0

  assert(v1.grad == 0.0)
  assert(v2.grad == 0.0)
  v3._backward()  
  # Gradient is multiplied by the value of the other node in the multiplication operation
  assert(v1.grad == 3.0)
  assert(v2.grad == 2.0)

def test_backward_pow():
  v1 = Value(2.0)
  v2 = v1 ** 3

  v2.grad = 2.0

  assert(v1.grad == 0.0)
  v2._backward()  
  # Local gradient comes from the Power rule from calculus (3*2)^2 
  assert(v1.grad == 72.0)

def test_topo_sort():
  v1 = Value(1.5)
  v2 = Value(1.0)
  v3 = v1 + v2

  v4 = Value(4.0)
  v5 = v3 * v4

  v6 = v5 ** 2

  topo_sorted = v6.topo_sort()
  # We haven't reversed the topo sorted list yet, so we expect v6 to be last element\
  # Note: There are multiple valid topo sort orderings possible, this is one. 
  assert(topo_sorted == [v4,v2,v1,v3,v5,v6])

def test_backward():
  v1 = Value(2.0)
  v2 = Value(0.5)

  v3 = v1 * v2

  v4 = Value(1.0)

  v5 = v3 + v4 

  v6 = v5 ** 2

  v6.backward()

  assert(v6.grad == 1.0)
  assert(v6.data == 4.0)
  
  assert(v5.grad == 4.0)
  assert(v5.data == 2.0)
 
  assert(v4.grad == 4.0)
  assert(v4.data == 1.0)

  assert(v3.grad == 4.0)
  assert(v3.data == 1.0)

  assert(v2.grad == 8.0)
  assert(v2.data == 0.5)

  assert(v1.grad == 2.0)
  assert(v1.data == 2.0)


if __name__=="__main__": 
  test_backward_add()
  test_backward_mul()
  test_backward_pow()
  test_topo_sort()
  test_backward()
