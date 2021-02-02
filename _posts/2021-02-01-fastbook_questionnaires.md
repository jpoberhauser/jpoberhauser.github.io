## Chapter 4 MNIST Basics

* https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb



1. How is a grayscale image represented on a computer? How about a color image?

A greyscale image is a MxN matrix, where M and N represent the height and width of the image. Each value in the matrix is the color value (in the greyscale case its just the intensity of black for example). 

A color image is pretty much the same except its represented by 3 matrices, one for each color channel in the RGB scheme (Red, Green, Blue)



2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?


3. Explain how the "pixel similarity" approach to classifying digits works.

This approach is where you take all your samples of "7"s for example, you line up all their matrices (in this case 28x28), and then you take the average of all pixels in the same position.
This approach gives you what an "average 7" looks like, and you can do the same for all numbers in your MNIST dataset.

Once you have your "average" of all the numbers, you can then do "predictions" by taking an image in the validation set, comparing to each of the average numbers you have (one for each class) and then returning the class for which your sample had the least amount of differences. 



4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

```
list_of_nums = [1, 3, 2, 4, 6, 8, 7]
res = [x*2 for x in list_of_nums if x%2 == 0]
```

5. What is a "rank-3 tensor"?
```
The rank of a tensor is the length of the tensors shape. So if you do `x.shape` and you get `torch.Size([784, 50])` as a result, thats a rank 2 tensor. 

So a rank-3 tensor should return something like `torch.Size([784, 50, 32])` for example. 
```

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?

The rank of a tensor is the length of the tensors shape


7. What are RMSE and L1 norm?

RMSE = Root Mean Squared error.

The formula for RMSE is 

RMS Errors= $\sqrt{\frac{\sum_{i=1}^n (\hat{y_i}-y_i)^2}{n}}$

How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
What is broadcasting?
Are metrics generally calculated using the training set, or the validation set? Why?
What is SGD?
Why does SGD use mini-batches?
What are the seven steps in SGD for machine learning?
How do we initialize the weights in a model?
What is "loss"?
Why can't we always use a high learning rate?
What is a "gradient"?
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
