import sys
import os

sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import boml as boml
import tensorflow as tf

# def test_meta_init_v2():
_input_1 = tf.placeholder(tf.float32, (None, 28, 28, 1))
_input_2 = tf.placeholder(tf.float32, (None, 84, 84, 3))
boml_meta_repr_v1= boml.BOMLNetMiniMetaReprV1(_input_2,use_T=True,use_Warp=True)
boml_meta_init_v2_omniglot = boml.BOMLNetOmniglotMetaInitV2(_input_1, dim_output=5)
boml_meta_init_v2 = boml.BOMLNetMiniMetaInitV2(_input_2, dim_output=5)
boml_meta_repr_v2_omniglot = boml.BOMLNetOmniglotMetaReprV2(_input_1)
boml_meta_repr_v2 = boml.BOMLNetMiniMetaReprV2(_input_2)


print(boml_meta_init_v2.out)
print(boml_meta_init_v2.re_forward().out)
print(boml_meta_init_v2_omniglot.out)
print(boml_meta_init_v2_omniglot.re_forward().out)
print(boml_meta_repr_v2.out)
print(boml_meta_repr_v2_omniglot.out)
print(boml_meta_repr_v2_omniglot.re_forward().out)
print(boml_meta_repr_v2.re_forward().out)


'''
def test_meta_repr_v2():
    _input_1 = tf.ones(dtype=tf.float32, shape=(5, 28, 28, 1), name='input_28_28')
    _input_2 = tf.ones(dtype=tf.float32, shape=(5, 84, 84, 3), name='input_84_84')
    boml_meta_repr_v2_omniglot = boml.BOMLNetMiniMetaReprV2(_input_2, dim_output=5)
    boml_meta_repr_v2 = boml.BOMLNetMiniMetaReprV2(_input_2, dim_output=5)
    with tf.Session() as sess:
        print(sess.run(boml_meta_repr_v2_omniglot.out))
        print(sess.run(boml_meta_repr_v2.out))
'''