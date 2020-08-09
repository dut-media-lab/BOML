try:
    from boml.optimizers.BOMLOpt import BOMLOpt
    from boml.optimizers.BOMLOptSGD import BOMLOptSGD
    from boml.optimizers.BOMLOptMomentum import BOMLOptMomentum
    from boml.optimizers.BOMLOptAdam import BOMLOptAdam
except ImportError:
    print("InnerOpt package not complete")
