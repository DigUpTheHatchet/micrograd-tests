class Value:
  """ stores a single scalar value and its gradient """

  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    self.prev = set(_children)
    self._op = _op # the op that produced this node (useful for graphviz / debugging / etc)
    self._backward = lambda: None
  
  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    result = self.data + other.data
    
    return Value(result, _children=(self,other), _op = '+')

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    result = self.data * other.data
    
    return Value(result, _children=(self,other), _op = '*')

  def __pow__(self, other):
    # Quick check that second argument is scalar valued. We're not supporting v1**v2 
    assert(isinstance(other,(float,int))) 
    result = self.data ** other

    return Value(result, _children=(self,), _op=f"**{other}")

  def __neg__(self): 
    return self * -1

  def __radd__(self, other):
    return self + other

  def __sub__(self, other): 
    return self + (-other)

  def __rsub__(self, other): 
    return (-other) + self

  def __rmul__(self, other): 
    return self * other

  def __truediv__(self, other): 
    return self * (other ** -1)

  def __rtruediv__(self, other): 
    return other * (self ** -1)


