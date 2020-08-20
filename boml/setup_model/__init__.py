try:
    from boml.setup_model.BOMLNet import BOMLNet
    from boml.setup_model.BOMLNetFeedForward import BOMLNetFeedForward
    from boml.setup_model.BOMLNetMetaInitV1 import BOMLNetMetaInitV1, BOMLNetMiniMetaInitV1, BOMLNetOmniglotMetaInitV1
    from boml.setup_model.BOMLNetMetaInitV2 import BOMLNetMiniMetaInitV2, BOMLNetOmniglotMetaInitV2
    from boml.setup_model.BOMLNetMetaReprV2 import BOMLNetMiniMetaReprV2, BOMLNetOmniglotMetaReprV2
    from boml.setup_model.BOMLNetMetaReprV1 import BOMLNetMetaReprV1, BOMLNetMiniMetaReprV1, BOMLNetOmniglotMetaReprV1

except ImportError:
    print("Networks Package not complete, check your file again.")
