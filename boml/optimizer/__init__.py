try:
    from boml.optimizer.BOMLOpt import BOMLOpt
    from boml.optimizer.BOMLOptSGD import BOMLOptSGD
    from boml.optimizer.BOMLOptMomentum import BOMLOptMomentum
    from boml.optimizer.BOMLOptAdam import BOMLOptAdam
except ImportError:
    print("InnerOpt package not complete")
