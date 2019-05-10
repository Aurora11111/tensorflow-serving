from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from utils import label_map_util
from utils import visualization_utils as vis_util

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_shape

import os
import glob
from PIL import Image
import numpy as np
import scipy
import time
from numpy import array

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS
CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def main(_):
  print(time.ctime())
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  IMAGE_NAME = '/home/rice/tensorflow1/models/research/object_detection/test_image/'

  TEST_IMAGE_PATHS = glob.glob(os.path.join(IMAGE_NAME, '*.*'))


  for image_path in TEST_IMAGE_PATHS:
    print(image_path)

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    print(image_np_expanded.shape)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'saved_model'
    request.model_spec.signature_name = 'serving_default'
    # request.inputs['inputs'].CopyFrom(
    #     tf.contrib.util.make_tensor_proto(image_np_expanded,shape=[1]))
    # print(time.ctime())
    # request.inputs['inputs'].CopyFrom(
    #     make_tensor_proto(image_np_expanded)
    # )

    # dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
    # tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_UINT8,
        tensor_shape=tensor_shape.as_shape(image_np_expanded.shape).as_proto(),
        # string_val=[open(image_path, 'rb').read()]
    )
    tensor_proto.tensor_content = image_np_expanded.tostring()

    request.inputs['inputs'].CopyFrom(tensor_proto)

    print(time.ctime())
    start = time.time()
    result = stub.Predict(request, 100.0)  # 10 secs timeout
    #print(result)
    print(time.time() - start)


    boxes = (result.outputs['detection_boxes'].float_val)
    classes = (result.outputs['detection_classes'].float_val)
    scores = (result.outputs['detection_scores'].float_val)

    box_np_arr = array([boxes[x:x + 4] for x in range(0, len(boxes), 4)])
    class_np_arr = array([classes[x:x + 1] for x in range(0, len(classes), 1)])
    score_np_arr = array([scores[x:x + 1] for x in range(0, len(scores), 1)])

    # print(type(box_np_arr),len(box_np_arr))
    # print(type(class_np_arr),len(classes))
    # print(type(class_np_arr),len(scores))

    # print(result)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=14, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_vis,result_box = vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(box_np_arr),
      np.squeeze(class_np_arr).astype(np.int32),
      np.squeeze(score_np_arr),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8,
      min_score_thresh=0.80)

    # Save inference to disk
    scipy.misc.imsave('%s.jpg' % (image), image_vis)

    for box,label_score in result_box.items():
        print(label_score,',',box)

    break


if __name__ == '__main__':
  tf.app.run()
