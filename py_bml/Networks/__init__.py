try:
    from py_bml.Networks.BMLNet import BMLNet, BMLNetFeedForward
    from py_bml.Networks.BMLNetHO_v1 import BMLNetMiniHO_v1,BMLNetOmniglotHO_v1
    from py_bml.Networks.BMLNetHO_v2 import BMLNetMiniHO_v2, BMLNetOmniglotHO_v2
    from py_bml.Networks.BMLNetMetaRepr_v2 import BMLNetOmniglotMetaRepr_v2, BMLNetMiniMetaRepr_v2
    from py_bml.Networks.BMLNetMetaRepr_v1 import BMLNetMiniMetaRepr_v1, BMLNetOmniglotMetaRepr_v1
except ImportError:
    print("Networks Package not complete, check your file again.")
