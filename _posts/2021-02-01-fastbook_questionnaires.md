# Fastbook Chapter 4 MNIST Basics

* https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb



### Questionnaire


#### 1. How is a grayscale image represented on a computer? How about a color image?

A greyscale image is a MxN matrix, where M and N represent the height and width of the image. Each value in the matrix is the color value (in the greyscale case its just the intensity of black for example). 

A color image is pretty much the same except its represented by 3 matrices, one for each color channel in the RGB scheme (Red, Green, Blue)



#### 2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?


#### 3. Explain how the "pixel similarity" approach to classifying digits works.

This approach is where you take all your samples of "7"s for example, you line up all their matrices (in this case 28x28), and then you take the average of all pixels in the same position.
This approach gives you what an "average 7" looks like, and you can do the same for all numbers in your MNIST dataset.

Once you have your "average" of all the numbers, you can then do "predictions" by taking an image in the validation set, comparing to each of the average numbers you have (one for each class) and then returning the class for which your sample had the least amount of differences. 



#### 4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

```
list_of_nums = [1, 3, 2, 4, 6, 8, 7]
res = [x*2 for x in list_of_nums if x%2 == 0]
```

#### 5. What is a "rank-3 tensor"?

```
The rank of a tensor is the length of the tensors shape. So if you do `x.shape` and you get `torch.Size([784, 50])` as a result, thats a rank 2 tensor. 

So a rank-3 tensor should return something like `torch.Size([784, 50, 32])` for example. 
```

#### 6. What is the difference between tensor rank and shape? How do you get the rank from the shape?

The rank of a tensor is the length of the tensors shape


#### 7. What are RMSE and L1 norm?

RMSE = Root Mean Squared error.

The formula for RMSE is 

RMS Errors= $\sqrt{\frac{\sum_{i=1}^n (\hat{y_i}-y_i)^2}{n}}$


L1 norm is a normalization technique to reduce the overall complexity of a model and avoid overfitting. L1 norm is applied to a model by adding a piece at the end of the loss function that penalized large parameters. 

L1 loss is simply `(a-b).abs().mean()`

#### 8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

There are a couple of techniques described in this chapter: one is broadcasting and the other is `einsum` or Einstein Summation. 


#### 9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

```
import torch
nums = torch.tensor([[1,2, 3],
                        [4,5, 6],
                        [7,8,9]])
nums = nums * 2
nums[1:, 1:]
```

#### 10. What is broadcasting?

* broadcasting is a technique used by packages like PyTorch or NumPy. The idea is that when you have operations on two objects that are different ranks, the package will so an **implicit** broadcasting where it makes the smaller ranked object into a larger ranked object so they are able to interact. 

From the fastai from the foundations course:

"When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when"

"If they are equal, or one of them is 1, in which case that dimension is broadcasted to make it the same size Arrays do not need to have the same number of dimensions. For example, if you have a 2562563 array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

Image (3d array): 256 x 256 x 3 Scale (1d array): 3 Result (3d array): 256 x 256 x 3

The numpy documentation includes several examples of what dimensions can and can not be broadcast together.


#### 11. Are metrics generally calculated using the training set, or the validation set? Why?

Metrics are calculated using the validation set. There is a difference between loss and metrics. 

Loss is what the model uses to calcualte gradients so the model knows which way to shift the weights. 

Metrics are generally calcualted during the validation so that the person training the model knows (in a human readbale way) how the model is doing.




#### 12. What is SGD?

Its stochastic gradient descent. This is how backpropagation works basically. 

The main idea is that we take a mini-batch of our training set, run it through the forward pass, then the backward pass where we take the gradients and update the weights. The algorithm does this until the specified number of times to run has been met. This optimizatoin algorithm updates weights by taking the loss function and seeing which way weights need to updated to minimize loss. 

#### 13. Why does SGD use mini-batches?

Batches are used for a couple of main reasons. GPUs work better with batches since all image batches will make easy parallelization. Another reason to use mini-batches is that they yield more stable gradients than running and updating a single image or data point at a time. 

#### 14. What are the seven steps in SGD for machine learning?

#### 15. How do we initialize the weights in a model?

Historically a popular initialization method was to use zeros or to initilizae at small numbers from a gaussian. Now, the standard is He initilization which works really well with the nonlinerarity that is most widely used at the moment `ReLU`. 

https://arxiv.org/abs/1502.01852


#### 16. What is "loss"?

Loss is a function that we use to tell the SGD step how far off the output of the forward pass was to the ground truth. There are many loss functions used in ML for both classification and regression problems. 

Some popular loss functions are MSE and cross-entropy loss. 


#### 17. Why can't we always use a high learning rate?

#### 18. What is a "gradient"?
Do you need to know how to calculate gradients yourself?
Why can't we use accuracy as a loss function?
Draw the sigmoid function. What is special about its shape?
What is the difference between a loss function and a metric?
What is the function to calculate new weights using a learning rate?
What does the DataLoader class do?
Write pseudocode showing the basic steps taken in each epoch for SGD.
Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
What does view do in PyTorch?
What are the "bias" parameters in a neural network? Why do we need them?
What does the @ operator do in Python?
What does the backward method do?
Why do we have to zero the gradients?
What information do we have to pass to Learner?
Show Python or pseudocode for the basic steps of a training loop.
What is "ReLU"? Draw a plot of it for values from -2 to +2.
What is an "activation function"?
What's the difference between F.relu and nn.ReLU?
The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
