from micrograd.value import Value

def test_relu():
  v1 = Value(122.0)
  v2 = v1.relu()
  assert(v2.data == 122.0)

  v2.grad = 1.0
  v2._backward()
  assert(v1.grad == 1.0)

  v3 = Value(0.0)
  v4 = v3.relu()
  assert(v4.data == 0.0)

  v4.grad = 1.0
  v4._backward()
  assert(v3.grad == 0.0)

  v5 = Value(-25.0)
  v6 = v5.relu()
  assert(v6.data == 0.0)

  v6.grad = 1.0
  v6._backward()
  assert(v5.grad == 0.0)


def test_tanh():
  v1 = Value(3.0)
  v2 = v1.tanh()
  assert(v2.data == 0.9950547536867305)
  
  v2.grad = 1.0
  v2._backward()
  assert(v1.grad == 0.009866037165440211)

  v3 = Value(0.0)
  v4 = v3.tanh()
  assert(v4.data == 0.0)

  v4.grad = 1.0
  v4._backward()
  assert(v3.grad == 1.0)

  v5 = Value(-25.0)
  v6 = v5.tanh()
  assert(v6.data == -1.0)

  v6.grad = 1.0
  v6._backward()
  assert(v5.grad == 0.0)

if __name__=="__main__": 
  test_relu()
  test_tanh()
