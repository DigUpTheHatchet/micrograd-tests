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


  