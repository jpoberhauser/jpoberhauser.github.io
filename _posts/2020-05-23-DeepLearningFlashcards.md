# Deep Learning Flashcards

1. TOC 
{:toc}


## Probability

### Sum Rule

* If you have mutually exclusive events:

	`P(A or b) = P(A) + P(B)`
	
	If you have a die, what is the probability of rolling a 3 OR a 4?
	
	`P(3 or 4) = P(3) + P(4) = 1/6 + 1/6 = 2/6 = 1/3`
	
* If you dont have mutually exclusive events:

In this case you ahve to remove their overlap (the probability that both of these happen so as not to over count.

12 Roman coins, 6 are silver
8 Greek coins, 3 are silver.

`P(Silver or Roman) = P(Silver) + (Roman)` --> This would overcount the probabilities so we need:

`P(Silver or Roman) = P(Silver) + (Roman) - P(Silver and Roman)`

`P(Silver or Roman) = 9/20 + 12/20 - 6/20 = 15/20 = 0.75`

### Product Rule

`P(A and B) = P(A)P(B)`

* IF **independent events** (meaning the probability of being female does not affect the probability of being brow-eyed and vice-versa)

`P(female and brown-eyed) = P(female) * P(brown-eyed)`

`P(female and brown-eyed) = 0.5 * 0.8 = 0.4`

* IF **NOT independent events**:

    `P(A and B) = P(B|A)P(A)`





	
**mental mode**l: how do you decide if you have to use product rule or sum rule (and vs or)?

	* if you have OR, then the probabilites must be summed since you are increasing the event space where you  can be right.
	
	* If you have AND, (this usually makes things less likely since two things have to happen instead of one. So here you multiply probabilities. 


## Panoptic Segmentation 

[Paper link](https://arxiv.org/abs/1801.00868)

We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). 


## Transformers

[Paper link](https://arxiv.org/pdf/1706.03762.pdf)

Architecture that proves that attention-only mechanisms (without RNNs like LSTMs) can improve on the results in translation or seq2seq tasks. 


## Attention

[Paper link](https://arxiv.org/pdf/1706.03762.pdf)

The attention mechanism looks at an input sequence and decides at each step which other parts of the sequence are most "important". This information gets passed along in translation encode-decoders to help the seq2seq model. 

## Deep Learning Regularization Methods

  1. dropout
  2. weight decay (L2 regularizatoin)
  3. data augmentation
  4. batch normalization
  5. label smoothing

## Test Time Augmentation

at **inference time**:
  1. do several transformations on the image
  2. Classify all those images
  3. Take average of predictions and use tha as final prediction
  
## NMS (non-maximum suppression)
* https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH

* Technique used in some object detectors to remove duplicate boxes. Many detectors generate anchors or region proposals and then need a way to remove boxes that are essentially finding the same thing. The algorithm is basically:

For a given class:

  input: list of several proposal boxes `B`, corresponding confidence scores `S` and overlap threshold `N`.
  Output: list of filtered proposals `D`
  Algorithm:
    1. Select proposal with highest confidence score, remove it from `B` and add it to proposal list `D`.
    2. Compare that proposal with all others, calculate IoI to all other proposals. If IoU is greater than threshold `N`, remove proposal from `B`.
    3. Again, take proposal with highest confidence for `B` and put it in `D`.
    4. Again, calculate IoU to all other proposals, eliminatie boxes w/ IoU higher than `N` from `B`.
    5. Repeat
    
    
## Label Smoothing

[paper](https://arxiv.org/pdf/1902.04103.pdf)

This modifies the hard ground truth labeling in cross entropy loss. 

Detection layers will usually compute a probability distribution over all classes with the softmaz function (from the raw outputs of the model). With **cross entropy** we usually compare a one-hot encoded distribution where the correct class has probability one and all others have 0. 

Label smoothing was proposed by Szegedy et al. as a form of regularization. This technique reduces the model’s confidence, measured by the difference between the largest and smallest logits.

## Discriminative Learning Rates

* source: fast.ai

* In transfer learning and Fine-tuning, we give the first layers, the midde laters, and the final layers different learning rates. 

 * The first layers, (more general patterns) are close to a local min already, so went a small lr to not move around too much. 
 
 * The later final layes have not been trained a lot on your specific task, so alerger lr is needed to get close to the minimum. 

## PCA

* dimensionality reduction algorithm. 

1. Find directoins of maximum varianse in data (orthogonal to each other)

2. Transform features onto directions of maxiumum variance

or another way to look at it: Covariance matrix --> Eigen decomposition --> sort by eigen values --> keep eigen vectors w/ highest eigen values

## Image Segmentation

* source fast.ai

* Each pixel has a unique class. We perform calssification for **every single pixel in the image**


## Deterministic (lagorithm, process, model, etc..)

* A model, procedure, algorithm etc, 

  * whose resulting behavior is **entirely** determind by its initial state and inputs, and which is not random or stochastic.
  
## Mixup (Data augmentation technique)

A data augmentation technique:

  * pick a random weight and take a weighted average of 2 images from your dataset. 
  
  * take a weighted average (with the same weights) of the 2 image labels
  
  * Create virtual training examples where you have an averaged $x_i$ and $x_j$ input vectors and $y_i$, $y_j$ one hot encoded labels. 
  
  * paper: https://arxiv.org/pdf/1710.09412.pdf


## FLOPS

Floating Point Operations (FLOPs)

Floating Point Operations per second (FLOPS)

## Encoder-Decoder Architectures

Input --> Encoder --> State --> Decoder --> Output

## Embeddings
 
* Multiplying by a one-hot encoded matrix, using the computational shortcut that is can be implemented by simply indexing directly.

* Categorical Embeddings for example: https://www.fast.ai/2018/04/29/categorical-embeddings/


## Cross-Entropy Loss (log loss)

source: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

binary classification:

$- (y * log(p) + (1 - y) log(1 - p))$

if class n > 2:


$-\sum_{c=1}^My_{o,c}\log(p_{o,c})$

```
def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)
```

## Softmax Activation Function

* ussually at the output layer and used for multi-class classification. 

* Compute for the normalized exponential function of all units in the layer. 

```
# calculate the softmax of a vector
def softmax(vector):
	e = exp(vector)
	return e / e.sum()
```

"squashes a vector of size k between 0 and 1. The sum of the whole vector equals 1. 


## CTC -  Connectionist Temporal Classification

* source: https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

Used in speech recognition, handwriting recognition, and other sequence problems. 

* a way to get around not knowing alignment between input and the output.

* To get the probability of an output given an input, CTC works by summing over the probability of all possible alignments between the two. 

## Latency

* The delay before a transfer of data begins following an isntructoin for its transfer

## Up-Convolutional Layer

`ConvTranspose2d` for example. 

* Suppose you have channels of 28x28 images and you need ti upsample to less chanlles of 56x56 images, this is the layer type you could use. A stride = 2 increases by a factor of 2. 

## Tensor Sizes - PyTorch

`[1, 1024, 14, 14]`

`[64, 3, 572, 572]`

`[batch_size, num_channels, width, height]`


## mAP - mean average precision

an overall view of the whole precision/recall curve. 

* Pick a confidence threshold so that recall is equal to each of `[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]` so for example you would pick thresholds like `[.9999, .98, .976, .96, .94, .93, .925, .85, .65, .5, .01 ]`

At each of those thresholds, calcualte **precision** and then average of all of them: so something you could get for your precision would be:
`[1, .99, .9, .85, .75, .7, .68, .65, .64, .6, .1]` and then take the average of this precision vector.


## Conv Net Architecure Patterns

Inout -> [[[Conv -> ReLU] * N] -> Pool] -> [FC -> ReLU] * k -> FC

## How should you normalize train/valid image sets?

* source fastbook, fast.ai

* You want to normalize train set with `train_mean` and `train_std` and also normalize the **validation** with `train_mean` and `train_std`.

Why?  If you had mostly green frogs in your train set and mostly red frogs in your validation set, ad you use their respective mean and std, you wouldnt be able to tell them apart and they would be on totally different units. 

## Transfer Learning (CNNs)

1. Replace the lat layer (the number of outputs needs to amtch the number of classes you want to predict)

2. Fine-tune the new layers

3. Fine tune the earlier layers with a small `lr` 

## Precision vs. Recall
 in words:
 
 **recall** "When the ground truth is **yes** how often does it predict **yes**". This is (in a simplistic view) a measure that is made worse by False Negatives.
 
 **precision** "When the model predicts **yes** how often is it correct?".  This is (in a simplistic view) a measure that is made worse by False Positives.

formulas: 

${\frac{tp}{tp+fn}}$

${\frac{tp}{tp+fp}}$



## What problem do ResNets solve?

* vanishing gradient

* With ResNets, the gradients can flow directly through the **skip connections** backwards from the later layers to initial filters. 

## Resnets

* Add simple trick of allowing skip connections (the possibility to lean idenitity functoins and skip layers that are not useful).

* Help with vanishing gradient and allows for deeper network architectures


## Update function for Backprop (without PyTorch):

```
for iteration in range(n_epochs):
	prediction = input.dot(weights)
	error = (goal_prediction - prediction) **2
	delta = goal_prediction - prediction
	weights = weights - (alpha * (input * delt)
```



$w_t = w_{t-1} - learningrate * {\frac{dL}{d_{wt-w}}$



```
def update(x, y, lr):
	wd = .0001
	y_hat = model(x)
	for p in model.parameters(): 
		w += p.sum()
	loss = loss_func(y_hat, y)
	loss.backward()
	for p in model.parameters():
		p.sub__(lr + p.grad)
		p.grad_zero()
```


## logs review

* logs answer the question: what power do we need to raise the base(10) to get the answer?

log_n(100) -> 10^x  =  100    x = 2


## Output size (convolutions)

O = [(i + 2p - k)  /  s  ]

where i =  input_size

p = paddings

k = kernel size

and s = stride

[] = "floor" function


## Initialization Techniques

Great papers:


* Understanding the difficulty of training deep feedforward neural networks

* Delving deep into Rectifiers

* All you need is a good init

* Exact solutions to the nonlinear dynamics of learning deep linear neural networks

* Fixup initializations

* self-normalizing neural networks 

## Basic Structure of NN in PyTorch

```
class myModel():
  def __init__(self, n_in, num_hidden, n_out):
    super().__init__()
    self.layers = [nn.Linear(n_in, num_hidden), nn.ReLU(), nn.Linear(num_hidden, n_out)]


  def __call__(self):
    for l in self.layers:
      x = l(x)
    return x
```

## yield in python

`yield` is a co-routine in python. You can have a function that doesnt return just one thing once, you can ask for it lots of times. When you call an iterator, it returns something that calls `next` on! It returns the next thing that is yielded. 

https://www.geeksforgeeks.org/use-yield-keyword-instead-return-keyword-python/

`iter()` generates the co-routine

`next()` gets the next thign that is yielded. 



## Train loop with and without Pytorch

 * fast.ai
 
 
 We can go from this: 
 
```
for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
#         set_trace()
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        loss = loss_func(model(xb), yb)

        loss.backward()
        with torch.no_grad():
            for l in model.layers:
                if hasattr(l, 'weight'):
                    l.weight -= l.weight.grad * lr
                    l.bias   -= l.bias.grad   * lr
                    l.weight.grad.zero_()
                    l.bias  .grad.zero_()
```
To this (which now includes validation) : 

```
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
#         print(model.training)
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
#         print(model.training)
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv
    
```




## minimal train loop: PyTorch

```
def fit():
    for epoch in range(epochs):
        for xb,yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
```
## minimal train and validation loop: PyTorch:

```
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
#         print(model.training)
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
#         print(model.training)
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv
```

## Training recipes: tips and tricks
Source: [A recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)

-   **visualize just before the net**. The unambiguously correct place to visualize your data is immediately before your `y_hat = model(x)` (or `sess.run` in tf). That is - you want to visualize _exactly_ what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.

-   **visualize prediction dynamics**. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.

-  **complexify only one at a time**. If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you’d expect. Don’t throw the kitchen sink at your model at the start. There are other ways of building up complexity - e.g. you can try to plug in smaller images first and make them bigger later, etc.


