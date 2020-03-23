# Paper Summary: A Disciplined Approach to NN hyper-parameters: Part1 - Learning rate, batch size, momentum, and weight decay


- [Original Paper](https://arxiv.org/pdf/1803.09820.pdf)

- Date: April 24th 2018

- Authors: Leslie N. Smith

Code examples and other info tidbits by @sgugger, taken from the link below:


- [Code](https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb)



 Part 1 includes : learning rate, batch size, momentum, and weight decay
 Part2 includes: architecture, regularization, dataset and task.

## Conext:

Related work: Cyclical Learning Rates for Training Neural Networks

- https://arxiv.org/pdf/1506.01186.pdf

Once cycle learning is dramatically better and faster. `learn.fit_one_cycle`


Getting the right learning rate is very important. If we choose one that is too small, we might never get to the right answer, as the model fails to convergence in a reasonable number of epochs. 


![LR is too small](img/lr_too_small.png "Too Small")


If we have the ideal LR, then the model converges quickly. 

![Ideal LR](img/ideal_lr.png "Ideal LR")

If the LR is too large, the model will fail to converge:

![LR is too large](img/lr_too_large.png "LR too large")


## Unfreezing, fine-tuning, and learning rates

Using pytorch's `fit_one_cycle` we can **fine tune**

1. We fitted 4 epochs

2. We added a few extra layers (linear) to the end, and we only trained the **new** laters we added. If we are bulding a model that is very similar, that is enough.

3. We should unfreeze more layers and train the whole model. 


## Intro 


The process of setting hyper-parameters, including designing the network architecture, requires expetise and extensive trial and error. 

* So how can we ste these parametrs? Grid Search?

"A grid search or random search (Bergstra & Bengio,2012) of the hyper-parameter space is computationally expensive and time consuming."


## The unreasonable effectiveness of validation/test loss

**Remark1:**

. The test/validation loss is a good indicator of the network’s convergence and should be
examined for clues.

![Signs of Overfitting](img/val_train_error.png "Overfitting clues")


### Cyclical LR and super convergence

If the lr is to small, overfitting can occur. LR's help regularize the training, but if the lr is too large, the training will not converge. We can perform a grid search of short runs to dinf learning rates that converge or diverge, **but there is an easier way**


**Cyclical LRs (CLRs)**  and the Learning Rate Test (LRT) were propsed in 2015 as a recipe for choosing LR. 

To use CLRs, one specifies a minimum and maximum lr as a boundary and a step size. The **stepsize** is the number of iterations (or epochs) used for each step, and a **cycle** consists of two such steps - one in which the lr linearly increases from the min to the max, and one in which it linearly decreases. 

In the **LR range test**, training starts with a small lr which is slowly increased linearly throughout a pre-training run. "When starting with a small learning rate, the network begins to converge and, as the learning rate increases,
it eventually becomes too large and causes the test/validation loss to increaseand the accuracy to decrease."

**Remark 3. A general principle is: the amount of regularization must be balanced for each dataset and architecture.**

## Picking a batch size

**Remark4: The takeaway message of this Section is that the practitioner’s goal is obtaining the
highest performance while minimizing the needed computational time**

## The 1cycle policy

https://sgugger.github.io/the-1cycle-policy.html

the learning rate finder: Beegin to train the model while increasing the lr from a very low to a very large one. Stop it when the loss starts to get very high. Plot losses against lr and pick a value before the minimum. 

For example below: anything between 10-2 and 10-1

![LR Finder Results](img/lr_finder.png "LR Finder results")

This approach came from the paper described in this summary. 

The author recommends to do a **cycle** with two steps of equal lengths. One goes from a lower lr to a hiegher one and then back to the minimum. The max should be the value picked from the LR Finder method, and the lower one a value roguhly times times lower. 

The length of the cycle (in epochs) should be slightly less than the total number of epochs. 

What Leslie Smith observed during his experiments, is that during the middle of the cycle, the high learning rates will act as a regularization method, and keep the network from overfitting. 

### Example: For the classification problem we are building in pytorch:

![One cycle policy](img/losses_and_lr.png "Losses and LR")

In this example, notice how the training loss goes up every once in a while. Why would that happen? **We are using one cycle policy**

The lr goes up and then it goes down. Why is this a good idea?

As you get closer and closer the global minimum, you want your LR to decrease ( you are close to the right spot). 

## Main Takeaway

**Learning rate annealing:** Is the idea that we should decrease the lr as we get further into training. This idea has been around for a long time. (This is the second half of the LR plot)

BUT

the idea of increasing the LR is much newer (Leslie Smith). 
IDEA: If you graduallly **increase your lr** what tends to happen is that you can now avoid getting stuck at a local minimum (a small valley somehwere along the loss surface, but not globally optimal.)

So if we gradually increase it, it will jump into the valley and then leave it, once it starts decreasing again, it can reach the global minimum. (explore the whole function surface)


“The essence of this learning rate policy comes from the observation that increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect.” Smith



### Appendix:

One Cycle policy in pytorch:

```
def fit_one_cycle(learn:Learner, cyc_len:int,
    max_lr:Union[Floats,slice]=defaults.lr, moms:Tuple[float,float]=(0.95,0.85),
    div_factor:float=25., pct_start:float=0.3, wd:float=None,
    callbacks:Optional[CallbackList]=None, tot_epochs:int=None,
    start_epoch:int=1)->None:
    "Fit a model following the 1cycle policy."

    max_lr = learn.lr_range(max_lr)
    callbacks = listify(callbacks)
    callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms,
    div_factor=div_factor, pct_start=pct_start,
    tot_epochs=tot_epochs,start_epoch=start_epoch))

    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
```





