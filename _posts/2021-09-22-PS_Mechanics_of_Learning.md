# Mechanics of Learning


How do neural networks learn? In this post, I consolidate all of the snippets and useful tips that helped me understand neural networks, backpropagation, and other techniques that nn use to learn. The three sources that I used the most are:

1. Deep Learning with PyTorch by Eli Stevens, Luca Antiga, Thomas Viehmann (link here https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
2. Grokking Deep Learning by Andrew W. Trask
3. CS231n Winter 2016: Lecture 4: Backpropagation, Neural Networks 1 lecture by Andrej Karpathy. (Link: https://www.youtube.com/watch?v=i94OvYb6noo)


## Goal

* The  goal of using derivatives is to guide the model training loop in which directions the parameters of the model need to be updated. 

* We want to optimize the loss function with respect to the parameters using **gradient descent**

From grokking deep learning:

* "Given a shared error, the network needs to figure out which weights contributed (so they can be adjusted) and which weights did **not** contribute (so they can be left alone. 

* "Hey, if you want this node to be x amount higher, then each of these previous four nodes needs to be `x*weights_1_2` amoutn higher/lower, because these weights were amplifying the prediction by `weights_1_2` times." -p.120


```
loss_rate_of_change_b = 
    (loss_fn(model(t_u, w, b + delta), t_c) -    # If you increase b, you get 110 loss for example
    (loss_fn(model(t_u, w, b - delta), t_c) /    # If you decrease b, you get 95 loss for example
    2.0 * delta                                  # For this example the change is positive, so we need to decrease b 
    
    
b = b - learning_rate * loss_rate_of_change_b
```

## Definitions

The **gradient**: answers the question: "what is the effect of the weight on the loss function?"

* During training, we want to compute the individual derivatives of the loss with respect to each parameter and put them in a vector of derivatives. 


![Gradients Illustrated](/images/83295.png "Examples")


## Recipe


We need to use chain rule to compute the derivative of loss w.r.t to **its inputs * derivative of model w.r.t the parameter.** 

$d_{loss}/d_{parameter} = d_{loss}/d{modelOutput} * d_{modelOutput} / d_{parameter} $

* Normalizing inputs: 

    * "The weight and bias live in a differently scaled space. If this is the case, a learning rate that's large enough to meaningfully ipdate one will be so large as to be usntalble for the other." --p.119
    * By normalizing inputs, the gradients will now be of similar magnitude, as we can use a single learning rate for both parameters. 

In PyTorch, the gradient gets calcualted and updated using **autograd** . "given a forward expression, no matter how nested, PyTorch will automatically provide the gradient of that expression with respect to its input parameters. 



## PyTorch Code
