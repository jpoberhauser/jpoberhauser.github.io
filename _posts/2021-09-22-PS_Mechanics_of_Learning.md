# Mechanics of Learning


### Sources: 

* Deep Learning with PyTorch Ch.5

* https://www.youtube.com/watch?v=i94OvYb6noo]

* Grokking Deep Learning

## Goal

* The  goal of using derivatives is to guide the model training loop in which directions the parameters of the model need to be updated. 

* We want to optimize the loss function with respoect to the parameters using **gradient descent**



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
