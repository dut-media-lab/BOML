try:
    from boml.ul_problem.BOMLOuterGrad import BOMLOuterGrad
    from boml.ul_problem.BOMLOuterGradReverse import BOMLOuterGradReverse
    from boml.ul_problem.BOMLOuterGradSimple import BOMLOuterGradSimple
    from boml.ul_problem.BOMLOuterGradImplicit import BOMLOuterGradImplicit
    from boml.ul_problem.BOMLOuterGradForward import BOMLOuterGradForward
except ImportError:
    print("OuterOpt package not complete")
