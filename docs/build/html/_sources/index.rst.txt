.. BOML documentation master file, created by
   sphinx-quickstart on Mon Sep  7 09:30:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOML's documentation!
================================
**Configuration & Status**

.. image:: https://travis-ci.com/dut-media-lab/BOML.svg?branch=master
   :target: https://github.com/dut-media-lab/BOML
   :alt: build status

.. image:: https://codecov.io/gh/dut-media-lab/BOML/branch/master/graph/badge.svg
   :target: https://github.com/dut-media-lab/BOML
   :alt: codecov
	
.. image:: https://readthedocs.org/projects/pybml/badge/?version=latest
   :target: https://github.com/dut-media-lab/BOML
   :alt: Documentation Status
	
.. image:: https://img.shields.io/badge/license-MIT-000000.svg
   :target: https://github.com/dut-media-lab/BOML
   :alt: License
	
.. image:: https://img.shields.io/github/languages/top/dut-media-lab/BOML
   :target: https://github.com/dut-media-lab/BOML
   :alt: Language
	
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/dut-media-lab/BOML
   :alt: Code style: black

BOML is a modularized optimization library that unifies several ML algorithms into a common bilevel optimization framework. It provides interfaces to implement popular bilevel optimization algorithms, so that you could quickly build your own meta learning neural network and test its performance.

**Key features of BOML**

- **Unified bilevel optimization framework** to address different categories of existing meta-learning paradigms. 
- **modularized algorithmic structure** to integrate a variety of optimization techniques and popular methods.
- **Unit tests with Travis CI and Codecov** to reach 99% coverage, and following **PEP8 naming convention** to guarantee the code quality. 
- **Comprehensive documentations** using sphinx and **Flexible functional interfaces** similar to conventional optimizers to help researchers quickly get familiar with the procedures.

**Related Links**

* `Go to the project home page <https://github.com/dut-media-lab/BOML>`_
* `Download the latest code bundle <https://codeload.github.com/elseifn/BOML/zip/master>`_

**Optimization Routine**

The figure below illustrates the general optimization routine by organized modules in BOML.

.. image:: _static/img/optimization_routine.png
	:alt: Bilevel Optimization Routine
	:align: center

**Documentation**

.. toctree::
	:maxdepth: 2
	:caption: Getting Started
   
	installation
	example
	
.. toctree::
	:maxdepth: 2
	:caption: Core Modules of BOML

	modules
	built_in
	extension

.. toctree::
	:maxdepth: 2
	:caption: Additional Information
	
	references
	license

