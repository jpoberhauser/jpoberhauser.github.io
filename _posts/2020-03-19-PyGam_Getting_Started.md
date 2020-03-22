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


