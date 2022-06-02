# OPTAMI: OPTimization for Applied Mathematics and Informatics

## Table of Contents

- [OPTAMI: OPTimization for Applied Mathematics and Informatics](#optami-optimization-for-applied-mathematics-and-informatics)
  - [Table of Contents](#table-of-contents)
  - [1. About](#1-about)
  - [2. Supported algorithms](#2-supported-algorithms)
    - [2.1 First-order methods](#21-first-order-methods)
    - [2.2 Second-order methods](#22-second-order-methods)
    - [2.3 Third-order methods](#23-third-order-methods)
    - [2.4 Accelerated envelopes](#24-accelerated-envelopes)
  - [3. Citation](#3-citation)
  - [4. For contributors](#4-for-contributors)
    - [4.1 Criteria for contributed algorithms](#41-criteria-for-contributed-algorithms)
    - [4.2 Recommendations for contributed algorithms](#42-recommendations-for-contributed-algorithms)
  - [5. Basic tests](#5-basic-tests)
    - [5.1 Unit tests](#51-unit-tests)
    - [5.2 Universal tests](#52-universal-tests)
    - [5.3 Performance tests](#53-performance-tests)
  - [6. Rights](#6-rights)


## 1. About

This package is dedicated to high-order optimization methods. All the methods can be used similarly to standard PyTorch optimizers.

## 2. Supported algorithms

Altough the development of this library was motivated primarily by the need in implementations of high-order optimization methods, we call contributors to commit methods of any order, and also already provide some of first-order methods in this library. Below we list all the currently supported algorithms divided into categories by their order, with the links on their source papers and/or their wiki pages.

### 2.1 First-order methods
* Similar Triangles method

   _Gasnikov, A.V., Nesterov, Y.E._ Universal Method for Stochastic Composite Optimization Problems. Comput. Math. and Math. Phys. **58**, 48–64 (2018). https://doi.org/10.1134/S0965542518010050

### 2.2 Second-order methods
* Damped Newton method

* Globally regularized Newton method

   _Mishchenko, K._ Regularized Newton method with global O(1/k^2) convergence. arXiv preprint arXiv:2112.02089 (2021). https://arxiv.org/abs/2112.02089

* Cubic regularized Newton method

   _Nesterov, Y., Polyak, B._ Cubic regularization of Newton method and its global performance. Math. Program. **108**, 177–205 (2006). https://doi.org/10.1007/s10107-006-0706-8

* Proximal Point Segment Search (Superfast) method

   _Nesterov, Y._ Superfast second-order methods for unconstrained convex optimization. Journal of Optimization Theory and Applications **191**, 1–30 (2021). https://doi.org/10.1007/s10957-021-01930-y

### 2.3 Third-order methods
* Basic tensor method (Bregman distance gradient method for p = 3)

   _Nesterov, Y._ Superfast second-order methods for unconstrained convex optimization. Journal of Optimization Theory and Applications **191**, 1–30 (2021). https://doi.org/10.1007/s10957-021-01930-y

### 2.4 Accelerated envelopes
* Superfast method

   _Nesterov, Y._ Superfast second-order methods for unconstrained convex optimization. Journal of Optimization Theory and Applications **191**, 1–30 (2021). https://doi.org/10.1007/s10957-021-01930-y

* Hyperfast method

   _Kamzolov D._ Near-optimal hyperfast second-order method for convex optimization. International Conference on Mathematical Optimization Theory and Operations Research, 167–178 (2020). https://doi.org/10.1007/978-3-030-58657-7_15

## 3. Citation

TBA

## 4. For contributors
### 4.1 Criteria for contributed algorithms
1. Class describing the algorithm (we denote it by `Algorithm`) is derived from torch.optim.optimizer.Optimizer
2. The paper introducing algorithm and the list of contributors are presented in docstring for `Algorithm`
3. The only required argument for constructor `Algorithm::__init__` is `params`
4. `Algorithm` does not takes the `model` itself in any way, only its `model.parameters()` as a `param` argument of constructor. As well as `Algorithm` does not take any information about loss, problem or other entities from outside. In other words, algorithms can use only zero-, first-, second- etc. oracle information provided by `closure` function, described below, or by the content of `grad` field of parameter `p`
5. All the necessary constants (from Lipschitz, Hölder, or Polyak–Łojasiewicz etc. condition) are the arguments of `Algorithm::__init__`, are provided with reasonable default value (working for the _Basic tests_) and have corresponding check raising `ValueError` if value is incorrect
6. Constructor `Algorithm::__init__` takes non-required boolean argument `verbose` controlling all the printing in stdout may be produced by `Algorithm`
7. Overridden method `Algorithm::step` takes one required parameter `closure`, that is the function evaluating loss (with a proper PyTorch forwarding) and that takes non-required boolean argument `backward` (if it is True, `closure` automatically performs backpropagation)
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
* Performance tests

### 5.1 Unit tests
Unit tests are implemented using the python `unittest` library, and are provided together with the source code of every algorithm in a distinct file. E.g., if algorithm is implemented in `algorithm.py`, unit tests are implemented in `test_algorithm.py` in the same directory. We ask contributors to provide their own versions of unit tests for the contributed algorithms. All the unit tests presented in library can be launched manually with a command `./run_unit_tests.sh`.

### 5.2 Universal tests

Universal tests check the expected behaviour and minimal performance requiremences for the algorithms on some toy problems. The main goal of these tests is to check the guarantees provided by the methods and eliminate the divergence of the algorithms. The universal tests are not available on edit for the side contributor, but can be complicated by authors in order to provide some more strong guarantees (for example, by checking the convergence rate on the problems with the known solution). In these cases, some algorithms that did not passed the enhanced tests may be deleted from _main_ branch until the correction (so we recommend to use only release versions of out library as a dependency in your project). All the universal tests presented in library can be launched manually with a command `./run_universal_tests.py`.

Now, the list of the used toy problems is as follows:
* a9a dataset (n = 123, m = 32561)

### 5.3 Performance tests

![plots_a9a_iters](/figure_iters.jpg?raw=true "Performance of all methods on a9a (a)")
![plots_a9a_time](/figure_time.jpg?raw=true "Performance of all methods on a9a (b)")

## 6. Rights
Copyright © 2020–2022 Dmitry Kamzolov
