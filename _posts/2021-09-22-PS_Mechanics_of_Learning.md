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




## PyTorch Code
