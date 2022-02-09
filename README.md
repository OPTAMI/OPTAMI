# OPTAMI – OPTimization for Applied Mathematics and Informatics

## Table of Contents

- [OPTAMI – OPTimization for Applied Mathematics and Informatics](#optami--optimization-for-applied-mathematics-and-informatics)
  - [Table of Contents](#table-of-contents)
  - [1. About](#1-about)
  - [2. Supported Optimizers](#2-supported-optimizers)
  - [3. Citation](#3-citation)
  - [4. For contributors](#4-for-contributors)
    - [4.1 Criteria for contributed algorithms:](#41-criteria-for-contributed-algorithms)
    - [4.2 Recommendations for contributed algorithms:](#42-recommendations-for-contributed-algorithms)
  - [5. Basic tests](#5-basic-tests)
  - [6. Rights](#6-rights)


## 1. About

This package is dediated to High-order optimization methods.
Methods can be used like basic PyTorch Optimizers.

## 2. Supported Optimizers

Now there are 5 main methods:
1. Hyperfast Second-Order Method (hyperfast.py)
   https://www.researchgate.net/publication/339390966_Near-Optimal_Hyperfast_Second-Order_Method_for_convex_optimization_and_its_Sliding
2. Superfast Second-Order Method (superfast.py)
   https://dial.uclouvain.be/pr/boreal/object/boreal%3A227146/
3. High-order BDGM (bdgm.py)
   https://dial.uclouvain.be/pr/boreal/object/boreal%3A227146/
4. Cubic Newton Method (cubic_newton.py)
   https://link.springer.com/content/pdf/10.1007/s10107-006-0706-8.pdf
5. Triangle Fast Gradient Method (TFGM.py)
   https://link.springer.com/content/pdf/10.1134/S0965542518010050.pdf

## 3. Citation

TBA

## 4. For contributors
### 4.1 Criteria for contributed algorithms:
1. Class describing the algorithm (we denote it by `Algorithm`) is derived from torch.optim.optimizer.Optimizer
2. The paper introducing algorithm and the list of contributors are presented in docstring for `Algorithm`
3. The only required argument for constructor `Algorithm::__init__` is `params`
4. `Algorithm` does not takes the `model` itself in any way, only its `model.parameters()` as a `param` argument of constructor. As well as `Algorithm` does not take any information about loss, problem or other entities from outside. In other words, algorithms can use only zero-, first-, second- etc. oracle information provided by `closure` function, described below, or by the content of `grad` field of parameter `p`
5. All the necessary constants (from Lipschitz condition, Hölder condition etc.) are the arguments of `Algorithm::__init__`, are provided with reasonable default value (working for the _Basic tests_) and have corresponding check raising `ValueError` if value is incorrect
6. Constructor `Algorithm::__init__` takes non-required boolean argument `verbose` controlling all the printing in stdout may be produced by `Algorithm`
7. Overridden method `Algorithm::step` takes one required parameter `closure`, that is the function evaluating loss (with a proper PyTorch forwarding) and that takes non-required boolean argument `backward` (if it is True, `closure` automatically performs backpropagation)
8. For the every `group` in `self.param_groups`, the commonly used variables (like constants approximations, counters etc.) are stored in `self.state[group['params'][0]]`
9. All the param-specific variables (typically x_k, y_k, z_k sequences) are stored by parts in `self.state[p]` for the corresponding `p` elements of `group['params']` (note, that it is very undesirable to storage anything in `self.state` in a form of List[Tensor] or Dict[Tensor], if it is possible to satisfy the prescribed requirement)
10. If `Algorithm` requires any additional functions for auxiliary calculations (excluding auxiliary optimization problems in need of iterative gradient-based subsolver), they are provided as a self-sufficient procedures before and outside the `Algorithm` implementation (note, that it is undesirable to use `@staticmethod` for this purpose)
11. Do not contribute several algorithms differing only in the usage of L-adaptivity, restarts procedure etc. If there is the special envelope in package implementing one of this extensions, make your `Algorithm` compatible with it. If there is not, add corresponding non-required boolean argument to `Algorithm::__init__`, controlling their usage. For backwards compatibility, if algorithm supports the compound usage with some envelope, add the corresponding non-required boolean argument anyway with default value `None` and further check that raises `AttributeError` if value is not `None`
12. Make sure that `Algorithm` passes _Basic tests_

### 4.2 Recommendations for contributed algorithms:
1. Make sure all the methods have clear comments
2. `Algorithm` and override methods should be provided with docstrings in [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
3. Try to use `@torch.no_grad()` annotation instead of `with torch.no_grad():` when it is possible
4. Class `Algorithm` should be named after the original name of the optimization algorithm (from its source paper), if it is unique and recognizable enough (like SARAH or Varag), or by the commonly accepted name of approach (like SimilarTriangles). The words "Method" and "Descent" should be omitted. Avoid the ambiguous abbreviations (e.g. use something like InterpolationLearningSGD instead of AMBSGD aka Accelerated Minibatch SGD)

## 5. Basic tests

Datasets info:
MNIST: dimension = 784, samples = 60000, L_3 > 0.5

## 6. Rights
Copyright (C) 2020 Dmitry Kamzolov
