from micrograd.value import Value

def test_add():
  v1 = Value(2.0)
  v2 = Value(3.0)

  v3 = v1 + v2

  assert(isinstance(v3, Value))
  assert(v3.data == 5.0)
  assert(v3._op == '+')
  assert(v3.prev == set((v1,v2)))

  v4 = Value(2.3)
  v5 = v4 + 1.7
  assert(isinstance(v5, Value))
  assert(v5.data == 4.0)
  

def test_mul():
  v1 = Value(1.5)
  v2 = Value(3)

  v3 = v1 * v2

  assert(isinstance(v3, Value))
  assert(v3.data == 4.5)
  assert(v3._op == '*')
  assert(v3.prev == set((v1,v2)))

  v4 = Value(10.0)
  v5 = v4 * 2.5
  assert(isinstance(v5, Value))
  assert(v5.data == 25.0)

def test_pow():
  v1 = Value(-2.0)

  v2 = v1**3

  assert(isinstance(v2, Value))
  assert(v2.data == -8.0)
  assert(v2._op == '**3')
  assert(v2.prev == set((v1,)))

def test_neg():
  v1 = Value(0.35)

  v2 = -v1

  assert(isinstance(v2, Value))
  assert(v2.data == -0.35)
  assert(v2._op == '*')
  assert(len(v2.prev) == 2)

def test_radd():
  v1 = Value(0.7555)

  v2 = 1.0 + v1

  assert(isinstance(v2, Value))
  assert(v2.data == 1.7555)
  assert(v2._op == '+')
  assert(len(v2.prev) == 2)

def test_sub():
  v1 = Value(100)

  v2 = v1 - 2.7

  assert(isinstance(v2, Value))
  assert(v2.data == 97.3)
  assert(v2._op == '+')
  assert(len(v2.prev) == 2)

def test_rsub():
  v1 = Value(3.0)

  v2 = 4.5 - v1

  assert(isinstance(v2, Value))
  assert(v2.data == -1.5)
  assert(v2._op == '+')
  assert(len(v2.prev) == 2)

def test_rmul():
  v1 = Value(3.5)

  v2 = 2 * v1

  assert(isinstance(v2, Value))
  assert(v2.data == 7.0)
  assert(v2._op == '*')
  assert(len(v2.prev) == 2)

def test_truediv():
  v1 = Value(9.0)
  v2 = Value(3.0)

  v3 = v1 / v2

  assert(isinstance(v3, Value))
  assert(v3.data == 3.0)
  assert(v3._op == '*')
  assert(len(v3.prev) == 2)

  v4 = Value(10.0)
  v5 = v4 / 2.0

  assert(isinstance(v5, Value))
  assert(v5.data == 5.0)
  assert(v5._op == '*')
  assert(len(v5.prev) == 2)

def test_rtruediv():
  v1 = Value(12.0)
  v2 = 36.0 / v1

  assert(isinstance(v2, Value))
  assert(v2.data == 3.0)
  assert(v2._op == '*')
  assert(len(v2.prev) == 2)


if __name__=="__main__": 
  test_add()
  test_mul()
  test_pow()
  test_neg()
  test_radd()
  test_sub()
  test_rsub()
  test_rmul()
  test_truediv()
  test_rtruediv()