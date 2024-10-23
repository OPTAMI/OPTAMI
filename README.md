# OPTAMI: OPTimization for Applied Mathematics and Informatics

## Installation
To install the package, run ```pip install OPTAMI```

## Table of Contents

- [OPTAMI: OPTimization for Applied Mathematics and Informatics](#optami-optimization-for-applied-mathematics-and-informatics)
  - [Table of Contents](#table-of-contents)
  - [1. About](#1-about)
  - [2. Supported Algorithms](#2-supported-algorithms)
    - [2.1 First-order Methods](#21-first-order-methods)
    - [2.2 Second-order Methods](#22-second-order-methods)
    - [2.4 Accelerated Envelopes](#23-accelerated-envelopes)
  - [3. Citation](#3-citation)
  - [4. For Contributors](#4-for-contributors)
    - [4.1 Criteria for Contributed Algorithms](#41-criteria-for-contributed-algorithms)
    - [4.2 Recommendations for Contributed Algorithms](#42-recommendations-for-contributed-algorithms)
  - [5. Basic Tests](#5-basic-tests)
    - [5.1 Unit Tests](#51-unit-tests)
    - [5.2 Universal Tests](#52-universal-tests)
  - [6. Rights](#6-rights)


## 1. About

This package is dedicated to second and high-order optimization methods. All the methods can be used similarly to standard PyTorch optimizers.

## 2. Supported Algorithms

Although the library is primarily focused on second-order optimization methods, we call contributors to commit methods of any order, and also already provide some of first-order methods in this library. Below we list all the currently supported algorithms divided into categories by their type and order, with the links on their source papers and/or their wiki pages.

### 2.1 First-order Methods

* **Gradient Descent**


* **Similar Triangles Method**

   _Gasnikov, A. and Nesterov, Y._ 2018. "Universal Method for Stochastic Composite Optimization Problems." Comput. Math. and Math. Phys. 58, pp.48–64. https://doi.org/10.1134/S0965542518010050

### 2.2 Second-order Methods
* **Damped Newton Method**


* **Cubic Regularized Newton Method**

   _Nesterov, Y. and Polyak, B._ 2006. "Cubic Regularization of Newton Method and its Global Performance." Mathematical Programming. 108, pp. 177–205. https://doi.org/10.1007/s10107-006-0706-8


* **Affine-Invariant Cubic Newton Method**

  _Hanzely, S., Kamzolov, D., Pasechnyuk, D., Gasnikov, A., Richtárik, P. and Takác, M._, 2022. "A Damped Newton Method Achieves Global $\mathcal O\left (\frac {1}{k^ 2}\right) $ and Local Quadratic Convergence Rate." Advances in Neural Information Processing Systems, 35, pp.25320-25334. https://proceedings.neurips.cc/paper_files/paper/2022/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html


* **Gradient Regularized Newton Method**

   _Mishchenko, K._, 2023. "Regularized Newton Method with Global $\mathcal O\left (\frac {1}{k^ 2}\right) $ Convergence." SIAM Journal on Optimization, 33(3), pp.1440-1462. https://doi.org/10.1137/22M1488752
    
    _Doikov, N. and Nesterov, Y._, 2024. "Gradient Regularization of Newton Method with Bregman Distances." Mathematical Programming, 204(1), pp.1-25. https://doi.org/10.1007/s10107-023-01943-7


* **Basic Tensor Method** (with Bregman Distance Gradient Method for $p = 3$)

   _Nesterov, Y._ 2021. "Implementable Tensor Methods in Unconstrained Convex Optimization." Mathematical Programming, 186, pp.157-183. https://doi.org/10.1007/s10107-019-01449-1

  _Nesterov, Y._ 2021. "Superfast Second-Order Methods for Unconstrained Convex Optimization." Journal of Optimization Theory and Applications, 191, pp.1-30. https://doi.org/10.1007/s10957-021-01930-y

### 2.3 Accelerated Envelopes
* **Nesterov Accelerated Tensor Method**

  _Nesterov, Y._ 2021. "Implementable Tensor Methods in Unconstrained Convex Optimization." Mathematical Programming, 186, pp.157-183. https://doi.org/10.1007/s10107-019-01449-1

  _Nesterov, Y._ 2021. "Superfast Second-Order Methods for Unconstrained Convex Optimization." Journal of Optimization Theory and Applications, 191, pp.1-30. https://doi.org/10.1007/s10957-021-01930-y


* **Nesterov Accelerated Tensor Method with A-Adaptation (NATA)**

  _Kamzolov, D., Pasechnyuk, D., Agafonov, A., Gasnikov, A. and Takáč, M._ 2024. "OPTAMI: Global Superlinear Convergence of High-order Methods."  https://arxiv.org/abs/2410.04083


* **Near-Optimal Accelerated Tensor Method**

  _Bubeck, S., Jiang, Q., Lee, Y.T., Li, Y. and Sidford, A._ 2019. "Near-Optimal Method for Highly Smooth Convex Optimization." In Conference on Learning Theory, pp. 492-507. PMLR. https://proceedings.mlr.press/v99/bubeck19a.html

  _Gasnikov, A., Dvurechensky, P., Gorbunov, E., Vorontsova, E., Selikhanovych, D., Uribe, C.A., Jiang, B., Wang, H., Zhang, S., Bubeck, S. and Jiang, Q._ 2019. "Near-Optimal Methods for Minimizing Convex Functions with Lipschitz $p$-th Derivatives." In Conference on Learning Theory, pp. 1392-1393. PMLR. https://proceedings.mlr.press/v99/gasnikov19b.html 

   _Kamzolov, D._ 2020. "Near-Optimal Hyperfast Second-order Method for Convex Optimization." International Conference on Mathematical Optimization Theory and Operations Research, pp. 167–178. https://doi.org/10.1007/978-3-030-58657-7_15


* **Near-Optimal Proximal-Point Acceleration Method with Segment Search**

   _Nesterov, Y._ 2021. "Inexact High-Order Proximal-Point Methods with Auxiliary Search Procedure." SIAM Journal on Optimization, 31(4), pp.2807-2828. https://doi.org/10.1137/20M134705X


* **Optimal Tensor Method**

   _Kovalev, D., Gasnikov, A._ 2022. "The First Optimal Acceleration of High-Order Methods in Smooth Convex Optimization." Advances in Neural Information Processing Systems, 35, pp.35339-35351. https://proceedings.neurips.cc/paper_files/paper/2022/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html


## 3. Citation
If you use code from OPTAMI, please cite both the original papers of the specific methods used and the following paper associated with the OPTAMI library: 
### _Kamzolov, D., Pasechnyuk, D., Agafonov, A., Gasnikov, A. and Takáč, M._ 2024. OPTAMI: Global Superlinear Convergence of High-order Methods. arXiv preprint arXiv:2410.04083.

## 4. For contributors
### 4.1 Criteria for contributed algorithms
1. Class describing the algorithm (we denote it by `Algorithm`) is derived from torch.optim.optimizer.Optimizer
2. The paper introducing algorithm and the list of contributors are presented in docstring for `Algorithm`
3. The only required argument for constructor `Algorithm::__init__` is `params`
4. `Algorithm` does not takes the `model` itself in any way, only its `model.parameters()` as a `param` argument of constructor. As well as `Algorithm` does not take any information about loss, problem or other entities from outside. In other words, algorithms can use only zero-, first-, second- etc. oracle information provided by `closure` function, described below, or by the content of `grad` field of parameter `p`
5. All the necessary constants (from Lipschitz, Hölder, or Polyak–Łojasiewicz etc. condition) are the arguments of `Algorithm::__init__`, are provided with reasonable default value (working for the _Basic tests_) and have corresponding check raising `ValueError` if value is incorrect
6. Constructor `Algorithm::__init__` takes non-required boolean argument `verbose` controlling all the printing in stdout may be produced by `Algorithm`
7. Constructor `Algorithm::__init__` takes non-required boolean argument `testing` which enables additional internal tests within the methods. Set `testing` to `True` for debugging and testing, or `False` to prioritize performance.
8. Overridden method `Algorithm::step` takes one required parameter `closure`, that is the function evaluating loss (with a proper PyTorch forwarding) and that takes non-required boolean argument `backward` (if it is True, `closure` automatically performs backpropagation)
8. For every `group` in `self.param_groups`, commonly used variables (like constants approximations, counters etc.) are stored in `self.state[group['params'][0]]`
9. All the param-specific variables (typically x_k, y_k, z_k sequences) are stored by parts in `self.state[p]` for the corresponding `p` elements of `group['params']` (note, that it is very undesirable to storage anything in `self.state` in a form of List[Tensor] or Dict[Tensor], if it is possible to satisfy the prescribed requirement)
10. If `Algorithm` requires any additional functions for auxiliary calculations (excluding auxiliary optimization problems in need of iterative gradient-based subsolver), they are provided as a self-sufficient procedures before and outside the `Algorithm` implementation (note, that it is undesirable to use `@staticmethod` for this purpose)
11. Do not contribute several algorithms differing only in the usage of L-adaptivity, restarts procedure etc. If there is the special envelope in package implementing one of this extensions, make your `Algorithm` compatible with it. If there is not, add corresponding non-required boolean argument to `Algorithm::__init__`, controlling their usage. For backwards compatibility, if algorithm supports the compound usage with some envelope, add the corresponding non-required boolean argument anyway with default value `None` and further check that raises `AttributeError` if value is not `None`
12. Make sure that `Algorithm` passes _Basic tests_
13. `Algorithm` must have static boolean attribute `MONOTONE` indicating whether method guarantees the monotonic decreasing of function value

### 4.2 Recommendations for contributed algorithms
1. Make sure all the methods have clear comments
2. `Algorithm` and override methods should be provided with docstrings in [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
3. Try to use `@torch.no_grad()` annotation instead of `with torch.no_grad():` when it is possible
4. Class `Algorithm` should be named after the original name of the optimization algorithm (from its source paper), if it is unique and recognizable enough (like SARAH or Varag), or by the commonly accepted name of approach (like SimilarTriangles). The words "Method" and "Descent" should be omitted. Avoid the ambiguous abbreviations (e.g. use something like InterpolationLearningSGD instead of AMBSGD aka Accelerated Minibatch SGD)

## 5. Basic tests

The basic tests are intended to check the correctness of contributed algorithms and benchmark them. These tests are launched automatically after the every update of the _main_ branch of repository, so we guarantee that implemented algorithms are correct and their performance non-decrease with the updates of implementations. Basic tests consist of three groups of tests:
* Unit tests
* Universal tests

### 5.1 Unit tests
Unit tests are implemented using the python `unittest` library, and are provided together with the source code of every algorithm in a distinct file. E.g., if algorithm is implemented in `algorithm.py`, unit tests are implemented in `test_algorithm.py` in the same directory. We ask contributors to provide their own versions of unit tests for the contributed algorithms. All the unit tests presented in library can be launched manually with a command `./run_unit_tests.sh`.

### 5.2 Universal tests

Universal tests check the expected behaviour and minimal performance requiremences for the algorithms on some toy problems. The main goal of these tests is to check the guarantees provided by the methods and eliminate the divergence of the algorithms. The universal tests are not available on edit for the side contributor, but can be complicated by authors in order to provide some more strong guarantees (for example, by checking the convergence rate on the problems with the known solution). In these cases, some algorithms that did not passed the enhanced tests may be deleted from _main_ branch until the correction (so we recommend to use only release versions of out library as a dependency in your project). All the universal tests presented in library can be launched manually with a command `./run_universal_tests.py`.

Now, the list of the used toy problems is as follows:
* a9a dataset (n = 123, m = 32561)

## 6. Rights
Copyright © 2020–2024 Dmitry Kamzolov
