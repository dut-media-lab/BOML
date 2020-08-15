try:
    from boml.networks.BOMLNet import BOMLNet, BOMLNetFeedForward
    from boml.networks.BOMLNetMetaInitV1 import BOMLNetMetaInitV1, BOMLNetMiniMetaInitV1, BOMLNetOmniglotMetaInitV1
    from boml.networks.BOMLNetMetaInitV2 import BOMLNetMiniMetaInitV2, BOMLNetOmniglotMetaInitV2
    from boml.networks.BOMLNetMetaReprV2 import BOMLNetMiniMetaReprV2, BOMLNetOmniglotMetaReprV2
    from boml.networks.BOMLNetMetaReprV1 import BOMLNetMetaReprV1, BOMLNetMiniMetaReprV1, BOMLNetOmniglotMetaReprV1

except ImportError:
    print("Networks Package not complete, check your file again.")
