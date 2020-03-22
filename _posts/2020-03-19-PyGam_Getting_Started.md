# pyGAM : Getting Started with Generalized Additive Models in Python


## Intro

I came across [pyGAM](https://github.com/dswah/pyGAM) a couple months ago, but found few examples online. Below is a more practical extension to the documentation found in the pyGAM homepage. 


Generalized additive models are an extension of generalized linear models. They provide a modeling approach that combines powerful statistical learning with interpretability, smooth functions, and flexibility. As such, they are a solid addition to the data scientist's toolbox.

This tutorial will not focus on the theory behind GAMs. For more information on that there is an amazing blog post by Kim Larsen [here](http://multithreaded.stitchfix.com/blog/2015/07/30/gam/)

Or for a much more in depth read check out Simon. N. Wood's great book, "Generalized Additive Models: an Introduction in R"
Some of the major development in GAMs has happened in the R front lately with the mgcv package by Simon N. Wood. At our company, we have been using GAMs with modeling success for structural modeling with machine learning, but needed a way to integrate it into our python-based "machine learning for production" framework.

## Installation

Installation is simple:


```bash
pip install pygam
```

pyGAM also makes use of scikit-sparse which you can install via conda. Not doing so will result in a warning and potential problems with the slowing down of optimization for models with monotonicity/convexity penalties.


## Classification Example

Let's go through a classification example...


We import `LogisticGAM` to begin the classification training process, and `load_breast_cancer` for the data. This data contains 569 observations and 30 features. The target variable in this case is whether the tumor of malignant or benign, and the features are several measurements of the tumor. For showcasing purposes, we keep the first 6 features only.

```python
import pandas as pd        
from pygam import LogisticGAM
from sklearn.datasets import load_breast_cancer
#load the breast cancer data set
data = load_breast_cancer()
#keep first 6 features only
df = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
target_df = pd.Series(data.target)
df.describe()
```


![Description of sample classification data](/images/describe_df.png "Description of sample classification data")



## Building the model

Since this is a classification problem, we want to make sure we use pyGam's `LogisticGAM()` function.

```python
X = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
y = target_df
#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)
```


The `summary()` function provides a statistical summary of the model. In the diagnostics below, we can see statistical metrics such as AIC, UBRE, log likelihood, and pseudo-R² measures.


![Model summary](/images/model_summary.png "Model summary")


To get the training accuracy we simply run

`gam.accuracy(X, y)`

And we get a .961335676625659 accuracy right off the bat. Obviously, this is training accuracy so there still needs to be a validation step to make sure we are not overfitting a model here. 


One of the nice things about GAMs is that their additive nature allows us to **explore and interpret individual features** by holding others at their mean. The snippet of code below shows these plots for the features included in the trained model. `generate_X_grid` helps us build a grid for nice plotting.

```python
XX = generate_X_grid(gam)
plt.rcParams['figure.figsize'] = (28, 8)
fig, axs = plt.subplots(1, len(data.feature_names[0:6]))
titles = data.feature_names
for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
    ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
    ax.set_title(titles[i])
plt.show()
```


![Partial dependency plots with confidence intervals](/images/plot_gam.png "Partial dependency plots with confidence intervals")



We can already see some very interesting results. It is clear that some features have a fairly simple linear relationship with the target variable. There are about three features that seem to have strong non-linear relationships though. We will want to combine the interpretability of these plots, and the power to prevent over fitting in GAMs to come up with a model that generalizes well to a holdout set of data.


---

**Partial dependency plots** are extremely useful because they are highly interpretable and easy to understand. For example at first examination we can tell that there is a very strong relationship between the mean radius of the tumor and the response variable. The bigger the radius of the tumor, the more likely it is to be malignant. Other features like the mean texture are harder to decipher, and we can already infer that we might want to make that a smoother line (we walk through smoothing parameters in the next section).



## Tuning Smoothness and Penalties


This is where the functionality of pyGAM begins to really shine through. We can choose to build a grid for parameter tuning or we can use intuition and domain expertise to find optimal smoothing penalties for the model.

Main parameters to keep in mind are:

`n_splines` , `lam` , and `constraints`


`n_splines` refers to the number of splines to use in each of the smooth function that is going to be fitted.

`lam` is the penalization term that is multiplied to the second derivative in the overall objective function.


`constraints` is a list of constraints that allows the user to specify whether a function should have a monotonically constraint. This needs to be a string in ['convex', 'concave', 'monotonic_inc', 'monotonic_dec','circular', 'none']


The default parameters that are being used in the model presented above are the following….
