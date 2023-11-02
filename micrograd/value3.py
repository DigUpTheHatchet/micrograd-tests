import math 

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
    result = Value(self.data + other.data, _children=(self,other), _op = '+')
    
    def _backward():
      self.grad += 1.0 * result.grad 
      other.grad += 1.0 * result.grad

    result._backward = _backward
    return result

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    result = Value(self.data * other.data, _children=(self,other), _op = '*')
    
    def _backward():
      self.grad += other.data * result.grad
      other.grad += self.data * result.grad

    result._backward = _backward
    return result

  def __pow__(self, other):
    # Quick check that second argument is scalar valued. We're not supporting v1**v2 
    assert(isinstance(other,(float,int))) 
    result = Value(self.data ** other, _children=(self,), _op=f"**{other}")

    def _backward():
      self.grad += (other * self.data) ** (other-1) * result.grad
      
    result._backward = _backward
    return result

  def relu(self):
    result = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (result.grad if result.data > 0 else 0)
    result._backward = _backward

    return result

  def tanh(self):
    e2x = math.exp(2 * self.data)
    tanhx = (e2x - 1)/(e2x + 1)   
    result = Value(tanhx, (self,), 'tanh')

    def _backward():
        self.grad += (1 - tanhx ** 2) * result.grad
    result._backward = _backward

    return result

  def backward(self):
    topo = self.topo_sort()
    self.grad = 1 # We will only ever call backward() on the final output node (self) so this is okay
    
    # go one value at a time and apply the chain rule to propogate the gradient backwards from the output
    for v in reversed(topo):
      v._backward()

  def topo_sort(self):    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for c in v.prev:
            build_topo(c)
        topo.append(v)
    
    build_topo(self)

    # topological order of all children nodes, and self at end of list
    return topo


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


