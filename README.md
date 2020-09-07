
# BOML - A Bilevel Optimization Library in Python for Multi-Task and Meta Learning
![Travis Status](https://travis-ci.com/dut-media-lab/BOML.svg?branch=master)
![codecov](https://codecov.io/gh/dut-media-lab/BOML/branch/master/graph/badge.svg)
![Documentation Status](https://readthedocs.org/projects/pybml/badge/?version=latest)
![license](https://img.shields.io/badge/license-MIT-000000.svg)
![Language](https://img.shields.io/github/languages/top/dut-media-lab/BOML)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

ReadMe.md file contains brief introduction to implement meta-initialization-based and meta-feature-based methods in few-shot learning field. Except for algorithms which have been proposed, various combinations of lower leve and upper level strategies are available. Moreover, it's flexible to build your own networks or use structures with attached documentation.

## Meta Learning 

Meta learning works fairly well when facing incoming new tasks by learning an initialization with favorable generalization capability. And it also has good performance even provided with a small amount of training data available, which gives birth to various solutions for different application such as few-shot learning problem.

We present a general bilevel optimization paradigm to unify different types of meta learning approaches, and the mathematical form could be summarized as below:<br>

<div align=center>
  
![Bilevel Optimization Model](https://github.com/dut-media-lab/BOML/blob/master/figures/p1.png)
</div>

## Generic Optimization Routine
Here we illustrate the generic optimization routine and hierarchically built strategies in the figure, which could be quikcly implemented in the following example.<br>

<div align=center>
  
![Hierarchically built strategies](https://github.com/dut-media-lab/BOML/blob/master/figures/p2.png)
</div>

## Running examples
```
from boml import utils
# initialize the BOMLOptimizer, specify strategies for ll_problem() and ul_problem()
boml_opt= boml.BOMLOptimizer('MetaInit', 'Simple', 'Simple')
#load dataset
dataset = boml.load_data.meta_omniglot(num_classes, (num_train, num_test))
ex = boml.BOMLExperiment(dataset)
# build network structure and initializer model parameters
meta_learner = boml_opt.meta_learner(ex.x, dataset, 'V1')
ex.model = boml_ho.base_learner(ex.x, meta_learner)
# lower objectives
loss_inner = utils.cross_entropy(ex.model.out, ex.y)
# define lower-level subproblem
inner_grad = boml_ho.ll_problem(loss_inner, lr, T, experiment=ex)
# define upper objectives and upper-level subproblem
loss_outer = utils.cross_entropy(ex.model.re_forward(ex.x_).out, ex.y_)
boml_ho.ul_problem(loss_outer, args.mlr, inner_grad,
                    meta_param=boml.extension.metaparameters())
# aggregate all the defined operations
boml_ho.aggregate_all()
```
## Documentation 
For more detailed information of basic function and construction process, please refer to our [Help Documentation](https://dut-media-lab.github.io/BOML/). Scripts in the directory named test_script are useful for constructing general training process.

Here we give recommended settings for specific hyper paremeters to quickly test performance of popular algorithms.

## Related Methods 
 - [Hyperparameter optimization with approximate gradient(HOAG)](https://arxiv.org/abs/1602.02355)
 - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks(MAML)](https://arxiv.org/abs/1703.03400)
 - [On First-Order Meta-Learning Algorithms(FOMAML)](https://arxiv.org/abs/1803.02999)
 - [Bilevel Programming for Hyperparameter Optimization and Meta-Learning(RHG)](http://export.arxiv.org/pdf/1806.04910)
 - [Truncated Back-propagation for Bilevel Optimization(TG)](https://arxiv.org/pdf/1810.10667.pdf)
 - [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace(MT-et)](http://proceedings.mlr.press/v80/lee18a/lee18a.pdf)
 - [Meta-Learning with warped gradient Descent(WarpGrad))](https://arxiv.org/abs/1909.00025)
 - [DARTS: Differentiable Architecture Search(DARTS)](https://arxiv.org/pdf/1806.09055.pdf)
 - [A Generic First-Order Algorithmic Framework for Bi-Level Programming Beyond Lower-Level Singleton(BA)](https://arxiv.org/pdf/2006.04045.pdf)


## License

MIT License

Copyright (c) 2020 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



