# Micrograd re-implementation

This repo is a learning resource to go along with [Andrej Karpathy's](https://karpathy.ai/) fantasic[youtube video](https://www.youtube.com/watch?v=VMj-3S1tku0) on [micrograd](https://github.com/karpathy/micrograd). Micrograd is a from scratch auto-grad implementation in Python.)

I was watching the 2.5 hour [video](https://www.youtube.com/watch?v=VMj-3S1tku0) and wondering what was the best way to engage with the content and make sure I was getting the most out of it. I was initially coding along as I watched the video, but I don't think this is a good way learn and retain information.

I decided it would be better to try to watch the whole thing and then try to re-implement micrograd myself, taking care to do as much as possible by myself and only peeking and Andrej's code when I absolutely had to.

For my re-implementation, I split the code up into five iterative steps: 
1. Implementing support for mathematical operators on the `Value` class
2. Implementing the details of the backward pass (backpropogation of gradients w.r.t output/loss)
3. Adding support for non-linear activation functions (ReLU and Tanh).
4. `Neuron`, `Layer` and `MLP` classes for putting together a very basic neural network (Multi-Layer Perceptron).
5. Additional functionality required to support iteratively training our basic neural network.

I've recorded the brief/instructions required to implement the code for each step in the `Tests`. 

Starting with step 1, you can follow along by creating your own re-implementation of micrograd (doing your best to avoid peeking at Andrej's code as much as possible).

Each step has an associated Pytest file that can be run to check on your implementation. When the tests are passing you know you can move onto the next step!

# Motivation

This approach was motivated by the [Nand to Tetris](https://www.nand2tetris.org/) course that I completed recently. 

In this course, the instructors took this approach:
* A large project is broken up into smaller sub-projects that successively build on top of one another.
* At each stage new material is presented and the student has to complete an implementation project to move onto the next chapter. 
* Runnable tests are provided at each stage so you know when your implementation is completed and you are ready to proceed to the next chapter.

I really enjoyed this approach to learning and wanted to see if I could apply it to my micrograd re-implementation :) 

# Bonus Questions

I've added some bonus questions [here](./Bonus%20Questions.md)

