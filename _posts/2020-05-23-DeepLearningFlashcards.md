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



