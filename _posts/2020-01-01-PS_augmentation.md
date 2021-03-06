# Paper Summary: Learning Data Augmentation Strategies for Object Detection



- [Original Paper](https://arxiv.org/pdf/1906.11172.pdf)

- Date: April 12th 2019

- Authors: Barret Zoph∗ , Ekin D. Cubuk∗ , Golnaz Ghiasi, Tsung-Yi Lin, Jonathon Shlens, Quoc V. Le, Google Brain


- [Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection)



## I. Abstract:

Data augmentation is a pretty important part of computer vision problems, and has been shown to improve image classification  (but hasnt been studied as much for object detection). This paper shows that augmentation techniques borrowed from classification work well, but the improvements are limited. Thus, they investigate how they can generate a **learned, specialized augmentation policy** specifically for object detection models. 

Experiments show that with a good augmentation policy you can get +2.3mAP. **Most importantly this learned strategy can be transferred to other detection datasets and models to achieve state of the art accuracy.**

This strategy is even better than state of the art regularizatoin methods for object detection. 


## II Introduction:

Recent exeperiments have shown that instead of manually designing augmentation strategies, learning a strategy from the data can imporve generalization. 

We consider 3 types of augmentation:

1. One that changes the colors and leaves bounding boxes unchanged

2. One that changes the entire image like shearing, translating, flipping in which the bounding box coordinates change.

3. One in which only the pixels _inside_ the bounding box are changed.


## III Methods:

They treat data augmentation search as a discrete optimization problem and optimize for generalization performance on the validation set. During training, on of the K (k=5) sub-policies will be selected at random and then applied to the current image. Each sub-policy has (N=2) image transformations. Each operatino is associated with 2 hyper-parameters (roation degree, tranlsation pixel count, brightness level, shearing angles, etc).


![Example policies ](/images/image_policies.png "5 Example Policies")


In preliminary expoeriments, they found 3 operations that are most beneficial to object detection:

1. **Color operations.** Distort color channels, without
impacting the locations of the bounding boxes (e.g.,
Equalize, Contrast, Brightness). 2

2. **Geometric operations.**  Geometrically distort the image, which correspondingly alters the location and
size of the bounding box annotations (e.g., Rotate,
ShearX, TranslationY, etc.).


3. **Bounding box operations.** Only distort the
pixel content contained within the bounding
box annotations (e.g., BBox Only Equalize,
BBox Only Rotate, BBox Only FlipLR).



The parameters for each operation were discretized into M uniformly spaced values. 

**Now finding th ebest policy becomes a search in discrete space** To search over 5 sub-policies the search space contains roughly $(22*6*6)^10 = 9.6*10^28$  which requires an efficient way to test through.

They chose to build a model by structuring the discretet optimization problem as the output soace of an RNN and employ reinformcement learning to update the weights of the model. 

The data: subset on COCO of 5k images, but found that good subpolicies generalized well to the full dataset. 

(Resnet50 backbone, and a RetinaNet detector with a cosine learning rate decay)

The reward signalfor the Reinforcement LEarning is the mAP on the holdout.



##  IIII Results:

Most commonly used operation in good policies was **Rotate** 

Two other commonly used operations are: **Equalize and bbox_only_equalize**


We notice that the final training loss of a detection model is lower when trained on a larger training set. When we apply the learned augmentation. the training loss is increased significantly for all


** Important**

The imporvmeents due to the learned augmentation is larger when the model is trained on smaller datasets. 

So as datasets get larger and larger, the relative marginal gain achieved by augmentation decreases. 

After 20,000 training images 


For 5,000 training samples, the learned aug can improve mAP by more that 70% relative to baseline. 

As the training set size increased, the effect of the learned augmentatoin policy is decreased, althoug improvements are still significant. 



"It is interesting to note that models trained with learned augmentation policy seem to do especially well on detecting
smaller objects, especially when fewer images are present in
the training dataset. 

For example, for small objects, applying the learned augmentation policy seems to be better than
increasing the dataset size by 50%. "

For
small objects, training with the learned augmentation policy
with 9000 examples results in better performance than the
baseline when using 15000 images. 

**In this scenario using our augmentation policy is almost as effective as doubling your dataset.**

## IV. What about other augmentation techniques?

We also find that other successful regularization techniques are not beneficial when applied in tandem with a
learned data augmentation policy. We carried out several
experiments with Input Mixup [52], Manifold Mixup [46]
and Dropblock [13]. For all methods we found that they
either did not help nor hurt model performance.

