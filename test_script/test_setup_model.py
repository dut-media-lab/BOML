import sys
import os

sys.path.append("../")
os.environ["DATASETS_FOLDER"] = "../"
os.environ["EXPERIMENTS_FOLDER"] = "../"
import boml as boml
import tensorflow as tf


def test_setup_model():
    # def test_meta_init_v2():
    _input_1 = tf.placeholder(tf.float32, (None, 28, 28, 1))
    _input_2 = tf.placeholder(tf.float32, (None, 84, 84, 3))
    boml_meta_repr_v1_t = boml.BOMLNetMiniMetaReprV1(
        _input_2, use_t=True, use_warp=True
    )
    boml_meta_repr_v1_warp = boml.BOMLNetMiniMetaReprV1(
        _input_2, use_t=True, use_warp=False, name="warp_test"
    )
    boml_meta_init_v1_mini = boml.BOMLNetMiniMetaInitV1(_input_2, dim_output=5)
    boml_meta_init_v2_omniglot = boml.BOMLNetOmniglotMetaInitV2(_input_1, dim_output=5)
    boml_meta_init_v2 = boml.BOMLNetMiniMetaInitV2(_input_2, dim_output=5)
    boml_meta_repr_v2_omniglot = boml.BOMLNetOmniglotMetaReprV2(_input_1)
    boml_meta_repr_v2 = boml.BOMLNetMiniMetaReprV2(_input_2)

    print(boml_meta_init_v1_mini.out)
    print(boml_meta_repr_v1_t.out)
    print(boml_meta_repr_v1_warp.out)
    print(boml_meta_init_v2.out)
    print(boml_meta_init_v2.re_forward().out)
    print(boml_meta_init_v2_omniglot.out)
    print(boml_meta_init_v2_omniglot.re_forward().out)
    print(boml_meta_repr_v2.out)
    print(boml_meta_repr_v2_omniglot.out)
    print(boml_meta_repr_v2_omniglot.re_forward().out)
    print(boml_meta_repr_v2.re_forward().out)
    print(boml.utils.get_rand_state(6))
    print(boml_meta_repr_v2.filter_vars("weights"))


if __name__ == "__main__":
    test_setup_model()
