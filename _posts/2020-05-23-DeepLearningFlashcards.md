# Deep Learning Flashcards

1. TOC 
{:toc}





## Panoptic Segmentation 

Source: https://arxiv.org/abs/1801.00868

We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). 


## Transformers

source: https://arxiv.org/pdf/1706.03762.pdf

Architecture that proves that attention-only mechanisms (without RNNs like LSTMs) can improve on the results in translation or seq2seq tasks. 


## Attention

source: https://arxiv.org/pdf/1706.03762.pdf

The attention mechanism looks at an input sequence and decides at each step which other parts of the sequence are most "important". This information gets passed along in translation encode-decoders to help the seq2seq model. 

## Deep Learning Regularization Methods

  1. dropout
  2. weight decay (L2 regularizatoin)
  3. data augmentation
  4. batch normalization

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
    
## Discrimitavie Learning Rates

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

$\text{Recall}}={\frac {tp}{tp+fn}}$

$\text{Precision}}={\frac {tp}{tp+fp}}$



## What problem do ResNets solve?

* vanishing gradient

* With ResNets, the gradients can flow directly through the **skip connections** backwards from the later layers to initial filters. 





