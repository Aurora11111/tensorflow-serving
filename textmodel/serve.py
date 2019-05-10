import tensorflow as tf
import os

SAVE_PATH = './trained_model_1531212474/checkpoints'
MODEL_NAME = 'model'
VERSION = 1
SERVE_PATH = './{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

tf.reset_default_graph()
# import the saved graph
saver = tf.train.import_meta_graph(checkpoint + '.meta')
variable_averages = tf.train.ExponentialMovingAverage(1)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:


    # get the graph for this session
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    # get the tensors that we need
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)

    model_inputx = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('input_x:0'))
    model_inputy = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('input_y:0'))
    model_dropout = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('dropout_keep_prob:0'))
    model_accuracy = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('accuracy/accuracy:0'))
    model_predictions = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('output/predictions:0'))

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_x': model_inputx,
                'input_y':model_inputy,

                'dropout_keep_prob':model_dropout
                },
        outputs={'output/predictions': model_predictions,
                 'accuracy/accuracy': model_accuracy
                 },
        method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # tensor_info_x = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('input_x:0'))
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('output/predictions:0'))

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'input_x': tensor_info_x},
    #         outputs={'output/predictions': tensor_info_y},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    legacy_init_op = tf.group(tf.tables_initializer(),
                              name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'predict_label': prediction_signature},
        legacy_init_op=legacy_init_op
    )
    # Save the model so we can serve it with a model server :)
    builder.save()




