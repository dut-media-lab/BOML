
import sys
import os
sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import boml as boml
import tensorflow as tf
#def test_meta_init_v2():
_input_1 = tf.ones(dtype=tf.float32, shape=(5, 28, 28, 1),name='input_28_28')
_input_2 = tf.ones(dtype=tf.float32, shape=(5, 84, 84, 3),name='input_84_84')
boml_meta_init_v2_omniglot = boml.BOMLNetOmniglotMetaInitV2(_input_1,dim_output=5)
boml_meta_init_v2 = boml.BOMLNetMiniMetaInitV2(_input_2,dim_output=5)
boml_meta_repr_v2_omniglot = boml.BOMLNetMiniMetaReprV2(_input_2)
boml_meta_repr_v2 = boml.BOMLNetMiniMetaReprV2(_input_2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    boml_meta_init_v2.initialize()
    print(sess.run(boml_meta_init_v2.out))
    boml_meta_init_v2_omniglot.initialize()
    print(sess.run(boml_meta_init_v2_omniglot.out))
    boml_meta_repr_v2.initialize()
    print(sess.run(boml_meta_repr_v2.out))
    boml_meta_repr_v2_omniglot.initialize()
    print(sess.run(boml_meta_repr_v2_omniglot.out))


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

