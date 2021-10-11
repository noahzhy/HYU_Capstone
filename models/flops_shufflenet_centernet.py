import os.path
import tempfile
import keras
from keras.models import *
from keras.layers import *
from keras.backend import *
from keras.optimizers import *
from keras.initializers import *

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential

from models.shufflenet_v2_modify import *


def count_flops(model):
    """ Count flops of a keras model
    # Args.
        model: Model,
    # Returns
        int, FLOPs of a model
    # Raises
        TypeError, if a model is not an instance of Sequence or Model
    """

    if not isinstance(model, (Sequential, Model)):
        raise TypeError(
            'Model is expected to be an instance of Sequential or Model, '
            'but got %s' % type(model))

    output_op_names = [_out_tensor.op.name for _out_tensor in model.outputs]
    sess = tf.keras.backend.get_session()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_op_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(os.path.join(tmpdir, 'graph.pb'))
        with tf.gfile.GFile(graph_file, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

        with tf.gfile.GFile(graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name='')
            tfprof_opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(new_graph, options=tfprof_opts)
            writer = tf.summary.FileWriter('gg', graph=new_graph)
            writer.flush()

    return flops


if __name__ == '__main__':
    # vgg = tf.keras.applications.vgg16.VGG16(
    #     include_top=True, weights=None,
    #     input_tensor=tf.keras.Input(batch_shape=(1, 224, 224, 3)))
    inputs = Input(shape=(224, 224, 3))
    model = shufflenetV2_centernet(Input(shape=(224, 224, 3)), [48, 96, 192, 1024])
    model.summary()

    flops = count_flops(model)
    print(flops)
