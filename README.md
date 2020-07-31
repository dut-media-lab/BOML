
# PyBML - A Bilevel Optimization Library in Python for Multi-Task and Meta Learning
![Travis Status](https://travis-ci.org/liuyaohua918/PyBML.svg?branch=master)
![Language](https://img.shields.io/badge/language-Python-green.svg)
PyBML provides interfaces to popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md file contains recommended instruction for training Maml-based and Meta-representation in few-shot learning field. It's flexible to build your own networks or use structures with attached documentation.
## Meta Learning and Multitask Learning

Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to new solutions to the few-shot learning problem.

In few-shot learning problems, we consider N-way K-shot classification tasks. For each iteration, $M$ batches of tasks $\{T_{i=1}^{M}\}$ are randomly sampled from distribution $P(T)$ associated with split datasets $D_{i}^{tr}$, $D_{i}^{val}$. Both datasets take the form of $D_{i}=\{(x_{i}^{k}, y_{i}^{k})\}^{K}_{k=1}$. And each K samples $x\in\mathcal{X}$ from the same class are bounded to specific label $y\in\mathcal{Y}$, which can be formulated as a mapping $h(x):\mathcal{X}\rightarrow \mathcal{Y}$. Meta learning aims to improve the model’s classification ability on new instances within N classes after training using sampled tasks. 

## Bilevel Structured Model 
Here We give definition of classical bilevel optimization problems.

Lower-Level Problem:
$$\theta^{*}:=\underset{\theta \in R^{n}}{\arg \min } \ell\left(\theta_{i}, \phi, D_{i}^{t r}, D_{i}^{v a l}\right)$$

To solve the LL(Lower-level) problem, we could either take advantage of the traing data or combine the extracted information from  UL(Upper Level) problem. Then optimizers performs gradient descent to optimize the task-specific parameters.

Upper-Level Problem:

$$\phi^{*}:=\underset{\theta \in R^{n}, \phi \in R^{m}}{\arg \min } F\left(\theta, \phi, D^{v a l}\right)$$

$$\text{ where } F\left(\theta, \phi, D^{val}\right)=\frac{1}{M} \sum_{i=1}^{M} \ell \left(\theta_{i}^{*}, \phi,D_{i}^{t r}, D_{i}^{val}\right)$$

To solve the UL(Upper Level) problem, various methods are implemented to compute the gradients of meta parameters using the validation data, which will be introduced in the next section.

## Related Algorithms 
 1. Management of Methods
    ![PyBML Models](https://github.com/liuyaohua918/PyBML/blob/master/figures/model.png)
 2. Related Papers
    - [Hyperparameter optimization with approximate gradient(Implicit HG)](https://arxiv.org/abs/1602.02355)
    - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(MAML)](https://arxiv.org/abs/1703.03400)
    - [On First-Order Meta-Learning Algorithms(FOMAML)](https://arxiv.org/abs/1803.02999)
    - [Bilevel Programming for Hyperparameter Optimization and Meta-Learning(Reverse HG)](http://export.arxiv.org/pdf/1806.04910)
    - [Truncated Back-propagation for Bilevel Optimization(Truncated Reverse HG)](https://arxiv.org/pdf/1810.10667.pdf)
    - [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace(MTNet)](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)
    - [Meta-Learning with warped gradient Descent(Warp-Grad))](https://arxiv.org/abs/1909.00025)
    - [DARTS: Differentiable Architecture Search(DARTS)](https://arxiv.org/pdf/1806.09055.pdf)
    - [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton(BDA)](https://arxiv.org/pdf/2006.04045.pdf)

## Package Structure
![Package Structure](https://github.com/liuyaohua918/PyBML/blob/master/figures/uml10.png)

## Documentation 

For more detailed information of basic function and construction process, please refer to our help page: [Help Documentation](https://liuyaohua918.github.io/PyBML/)

It's flexible to build your own networks or use structures in py_bm.networks. Scripts in the directory named train_script are useful for basic training process.

Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.


