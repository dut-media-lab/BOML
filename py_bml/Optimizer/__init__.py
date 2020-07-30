try:
    from py_bml.Optimizer.BMLOpt import BMLOpt
    from py_bml.Optimizer.BMLOptSGD import BMLOptSGD
    from py_bml.Optimizer.BMLOptMomentum import BMLOptMomentum
    from py_bml.Optimizer.BMLOptAdam import BMLOptAdam
except ImportError:
    print("InnerOpt package not complete")
