# 1. What is backpropagation used for in neural networks? 

# 2. What does the derivative f'(x) of a function (x) tell you?

# 3. What is the local gradient at a multiplication operation?
e.g. if `L = d*f``; What is `dL/dd`? (answer is: f) 

# 4. What is the local gradient at an addition operation?
e.g. if `d = c + e`; What is `dd/dc`? (answer is 1.0)

# 5. What is the chain rule? 
e.g if `dz/dx` = `dz/dy * dy/dx`

# 6. How do we backprop gradient through an addition node?
We just distribute the gradient back to both child nodes unchanged (because the local gradient at both child nodes is 1.0, and we don’t need to do the multiply)

# 7. How do we backprop through a multiplication node?
We multiply the gradient propagated back by the local gradient of the other child node of the multiplication node

# 8. If we want the output value to go up, in what direction should we nudge our inputs?
* nudge in direction of gradient = output goes up
* nudge in opposite direction of gradient = output goes down

# 9. What are some common nonlinear functions used as activation functions? 
tanh, sigmoid, relu 

# 10. If an input has a negative gradient, will we want to increase or decrease it’s weight?

# 11. What is a leaf node? What happens when we call `_backward()`` on a leaf node?

# 12. If `b = a + a``; what is `db/da`? 
(answer: 2.0)

# 13. What is the derivative of `e^x`? 
(answer: e^x)

# 14. How can you express division without explictly using a division operation? 
(a/b = a * b**-1)

# 15. What is the definition of the tanh function?

# 16. What is the derivative of the tanh function?

# 17. What is the definition of the ReLU function?

# 18. What is the derivative of the ReLU function?

# 19. What is the derivative of 5x^3? 
(answer: 15x^2) by the Power rule

# 20. What would happen if we did not randomly initialize the weights/biases of our MLP and left them all at 0.0?

Backpropogation would not work as all values/gradients would become 0.0

# 21. I have a weight with grad = -0.72. Which direction should I nudge the weight?
* If grad is negative, then this weight has a negative influence on the loss, so we should nudge/increase this weight up (so that the loss goes down).

# 22. Why do we not care about the computed gradients of the input values?


# 23. What would happen if we forget to zero the gradients of our weights/biases before each

Gradients would accumulate from one training epoch to the next, multiplying each time. This may actually be desired, check out the concept of **Momentum**.