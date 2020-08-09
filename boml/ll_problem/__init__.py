try:
    from boml.ll_problem.BOMLInnerGradTrad import BOMLInnerGradTrad
    from boml.ll_problem.BOMLInnerGradSimple import BOMLInnerGradSimple
    from boml.ll_problem.BOMLInnerGradAggr import BOMLInnerGradAggr
except ImportError:
    print('LLProblem package missing!')
