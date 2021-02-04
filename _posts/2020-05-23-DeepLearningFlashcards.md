# Deep Learning Flashcards

1. TOC 
{:toc}


## NMS (Non maximum supression)

* Technique used in some object detectors to remove duplicate boxes. Many detectors generate anchors or region proposals and then need a way to remove boxes that are essentially finding the same thing. The algorithm is basically:

For a given class:

1. Take a list of all boxes the detector finds along with the confidence scores. Take the box with the highest confidence and add it to the final list, while removing it from the original proposed list. 

2. Calculate the IoU with all other boxes in the proposed set and if the IoU with any of the proposed boxes and the one with the highest score is higher than the desired threshold (.7 for example), remove all those boxes. 

3. Take the box with the highest confidence from the proposed list and move it to the final list.

4. Calculate the IoU with all other boxes in the proposed set and if the IoU with any of the proposed boxes and the one with the highest score is higher than the desired threshold (.7 for example), remove all those boxes. 

5. Repeat until there are no more boxes in the proposed list. 


## Panoptic Segmentation 

Source: https://arxiv.org/abs/1801.00868

We propose and study a task we name panoptic segmentation (PS). Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assign a class label to each pixel) and instance segmentation (detect and segment each object instance). 


## Transformers

source: https://arxiv.org/pdf/1706.03762.pdf

Architecture that proves that attention-only mechanisms (without RNNs like LSTMs) can improve on the results in translation or seq2seq tasks. 


## Attention

source: https://arxiv.org/pdf/1706.03762.pdf

The attention mechanism looks at an input sequence and decides at each step which other parts of the sequence are most "important". This information gets passed along in translation encode-decoders to help the seq2seq model. 

