"""Functions to export object detection inference graph."""
import logging
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../slim"))
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
import tensorflow as tf

# Assuming object detection API is available for use
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file

slim = tf.contrib.slim


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when
# newer version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    clear_devices,
    initializer_nodes,
    optimize_graph=True,
    variable_names_blacklist=''):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    raise ValueError(
        'Input checkpoint "' + input_checkpoint + '" does not exist!')

  if not output_node_names:
    raise ValueError(
        'You must supply the name of a node to --output_node_names.')

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ''

  with tf.Graph().as_default():
    tf.import_graph_def(input_graph_def, name='')

    if optimize_graph:
      logging.info('Graph Rewriter optimizations enabled')
      rewrite_options = rewriter_config_pb2.RewriterConfig(
          optimize_tensor_layout=True)
      rewrite_options.optimizers.append('pruning')
      rewrite_options.optimizers.append('constfold')
      rewrite_options.optimizers.append('layout')
      graph_options = tf.GraphOptions(
          rewrite_options=rewrite_options, infer_shapes=True)
    else:
      logging.info('Graph Rewriter optimizations disabled')
      graph_options = tf.GraphOptions()
    config = tf.ConfigProto(graph_options=graph_options)
    with session.Session(config=config) as sess:
      if input_saver_def:
        saver = saver_lib.Saver(saver_def=input_saver_def)
        saver.restore(sess, input_checkpoint)
      else:
        var_list = {}
        reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
          try:
            tensor = sess.graph.get_tensor_by_name(key + ':0')
          except KeyError:
            # This tensor doesn't exist in the graph (for example it's
            # 'global_step' or a similar housekeeping element) so skip it.
            continue
          var_list[key] = tensor
        saver = saver_lib.Saver(var_list=var_list)
        saver.restore(sess, input_checkpoint)
        if initializer_nodes:
          sess.run(initializer_nodes)

      variable_names_blacklist = (variable_names_blacklist.split(',') if
                                  variable_names_blacklist else None)
      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.split(','),
          variable_names_blacklist=variable_names_blacklist)

  return output_graph_def



def _image_tensor_input_placeholder(input_shape=None):
  """Returns input placeholder and a 4-D uint8 image tensor."""
  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor


def _tf_example_input_placeholder():
  """Returns input that accepts a batch of strings with tf examples.
  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_tf_example_placeholder = tf.placeholder(
      tf.string, shape=[None], name='tf_example')
  def decode(tf_example_string_tensor):
    tensor_dict = tf_example_decoder.TfExampleDecoder().decode(
        tf_example_string_tensor)
    image_tensor = tensor_dict[fields.InputDataFields.image]
    return image_tensor
  return (batch_tf_example_placeholder,
          tf.map_fn(decode,
                    elems=batch_tf_example_placeholder,
                    dtype=tf.uint8,
                    parallel_iterations=32,
                    back_prop=False))


def _encoded_image_string_tensor_input_placeholder():
  """Returns input that accepts a batch of PNG or JPEG strings.
  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='encoded_image_string_tensor')
  def decode(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor
  return (batch_image_str_placeholder,
          tf.map_fn(
              decode,
              elems=batch_image_str_placeholder,
              dtype=tf.uint8,
              parallel_iterations=32,
              back_prop=False))


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
    _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
}


def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.
  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.
    * detection_masks: (Optional) float32 tensor of shape
      [batch_size, num_boxes, mask_height, mask_width] containing masks for each
      detection box.
  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'num_detections': [batch]
    output_collection_name: Name of collection to add output tensors to.
  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  label_id_offset = 1
  boxes = postprocessed_tensors.get('detection_boxes')
  scores = postprocessed_tensors.get('detection_scores')
  classes = postprocessed_tensors.get('detection_classes') + label_id_offset
  masks = postprocessed_tensors.get('detection_masks')
  num_detections = postprocessed_tensors.get('num_detections')
  outputs = {}
  outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
  outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
  outputs['detection_classes'] = tf.identity(classes, name='detection_classes')
  outputs['num_detections'] = tf.identity(num_detections, name='num_detections')
  if masks is not None:
    outputs['detection_masks'] = tf.identity(masks, name='detection_masks')
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])
  if masks is not None:
    tf.add_to_collection(output_collection_name, outputs['detection_masks'])
  return outputs



def _write_saved_model(saved_model_path,
                       trained_checkpoint_prefix,
                       inputs,
                       outputs):
  """Writes SavedModel to disk.
  Args:
    saved_model_path: Path to write SavedModel.
    trained_checkpoint_prefix: path to trained_checkpoint_prefix.
    inputs: The input image tensor to use for detection.
    outputs: A tensor dictionary containing the outputs of a DetectionModel.
  """
  saver = tf.train.Saver()
  with session.Session() as sess:
    saver.restore(sess, trained_checkpoint_prefix)
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

    tensor_info_inputs = {
          'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
    tensor_info_outputs = {}
    for k, v in outputs.items():
      tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

    detection_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
              inputs=tensor_info_inputs,
              outputs=tensor_info_outputs,
              method_name=signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'detection_signature':
                  detection_signature,
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  detection_signature,
          },
      )
    builder.save()

def _export_inference_graph(
                            detection_model,

                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            optimize_graph=True,
                            output_collection_name='inference_op'):
  """Export helper."""
  #tf.gfile.MakeDirs(output_directory)
  #frozen_graph_path = os.path.join(output_directory,
  #                                 'frozen_inference_graph.pb')
  #saved_model_path = os.path.join(output_directory, 'saved_model')
  saved_model_path = output_directory

  placeholder_args = {}

  placeholder_args['input_shape'] = input_shape
  placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](**placeholder_args)
  inputs = tf.to_float(input_tensors)
  preprocessed_inputs = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(preprocessed_inputs)
  postprocessed_tensors = detection_model.postprocess(output_tensors)
  outputs = _add_output_tensor_nodes(postprocessed_tensors,
                                     output_collection_name)
  # Add global step to the graph.
  slim.get_or_create_global_step()


  _write_saved_model(saved_model_path, trained_checkpoint_prefix,
                     placeholder_tensor, outputs)


def export_inference_graph(
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           optimize_graph=True,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None
                          ):
  """Exports inference graph for the model specified in the pipeline config.
  Args:
    input_type: Type of input for the graph. Can be one of [`image_tensor`,
      `tf_example`].
    pipeline_config: pipeline_pb2.TrainAndEvalPipelineConfig proto.
    trained_checkpoint_prefix: Path to the trained checkpoint file.
    output_directory: Path to write outputs.
    input_shape: Sets a fixed shape for an `image_tensor` input. If not
      specified, will default to [None, None, None, 3].
    optimize_graph: Whether to optimize graph using Grappler.
    output_collection_name: Name of collection to add output tensors to.
      If None, does not add output tensors to a collection.
    additional_output_tensor_names: list of additional output
    tensors to include in the frozen graph.
  """
  detection_model = tf.saved_model.builder.SavedModelBuilder(trained_checkpoint_prefix)

  _export_inference_graph( detection_model,
                          trained_checkpoint_prefix,
                          output_directory,
                          additional_output_tensor_names,
                          input_shape,
                          optimize_graph,
                          output_collection_name
                          )


if __name__ == '__main__':
    # Configuration for model to be exported
    config_pathname = 'training/faster_rcnn_inception_v2_pets.config'

    # Input checkpoint for the model to be exported
    # Path to the directory which consists of the saved model on disk (see above)
    trained_model_dir = './trained_model_1531212474/checkpoints'

    # Create proto from model confguration
    #configs = get_configs_from_pipeline_file(config_pathname)
    #pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

    # Read .ckpt and .meta files from model directory
    checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Model Version
    model_version_id = '1'

    # Output Directory
    output_directory = '/home/rice/PycharmProjects/mytensorflow_sample/server/' + str(model_version_id)

    # Export model for serving
    export_inference_graph(
                           trained_checkpoint_prefix=input_checkpoint,
                           output_directory=output_directory
                           )