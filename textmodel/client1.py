from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import time
import os
import json
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from data_helper import clean_str

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')

FLAGS = tf.app.flags.FLAGS

def numpy_array_int32(data):
  (im_width, im_height) = data.shape
  return np.array(data).reshape(
      (im_width,im_height)).astype(np.int32)

def numpy_array_float(data):
  (im_width, im_height) = data.shape
  return np.array(data).reshape(
      (im_width,im_height)).astype(np.float32)

def main(_):
  print(time.ctime())
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  #Send request
  params = json.loads(open('./parameters.json').read())
  checkpoint_dir = './trained_model_1531212474/'
  if not checkpoint_dir.endswith('/'):
      checkpoint_dir += '/'

  """Step 1: load data for prediction"""
  test_file = './data/data.csv'
  # test_examples = json.loads(open(test_file).read())
  test_examples = pd.read_csv(test_file, dtype={'data': object})
  df = pd.read_csv(test_file, dtype={'data': object})
  # selected = ['text','title','style','structural','tag']
  selected = ['data', 'tag']
  non_selected = list(set(df.columns) - set(selected))

  df = df.drop(non_selected, axis=1)  # Drop non selected columns
  df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
  df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

  # Map the actual labels to one hot labels
  labels = sorted(list(set(df[selected[1]].tolist())))
  one_hot = np.zeros((len(labels), len(labels)), int)
  np.fill_diagonal(one_hot, 1)
  label_dict = dict(zip(labels, one_hot))

  vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
  vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

  x_raw = df[selected[0]].apply(lambda x: clean_str(x)).tolist()
  x_test = [data_helper.clean_str(x) for x in x_raw]
  x_test = np.array(list(vocab_processor.transform(x_test)))
  logging.info('The number of x_test: {}'.format(len(x_test)))

  y_test = None
  y_raw = df[selected[1]].apply(lambda y: label_dict[y]).tolist()
  y_test = np.array(y_raw)
  logging.info('The number of y_test: {}'.format(len(y_test)))

  batches = data_helper.batch_iter(list(zip(x_test,y_test)), params['batch_size'], 1, shuffle=False)
  all_predictions = []
  start = time.time()


  for test_batch in batches:
    x_test_batch, y_test_batch = zip(*test_batch)

    x_test_batch = np.array(x_test_batch)
    x_test_batch = numpy_array_int32(x_test_batch)

    y_test_batch = np.array(y_test_batch)
    y_test_batch = numpy_array_float(y_test_batch)
    print(type(y_test_batch))

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'saved_model'
    request.model_spec.version.value = 1
    request.model_spec.signature_name = 'predict_label'

    request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto(x_test_batch))
    request.inputs['input_y'].CopyFrom(tf.contrib.util.make_tensor_proto(y_test_batch))
    print("lalalala")
    request.inputs['dropout_keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0))
    print("hahahahaha")

    result = stub.Predict(request, 10.0)  # 10 secs timeout

    print(result.outputs['accuracy/accuracy'])
    batch_predictions = (result.outputs['output/predictions'].int64_val)
    all_predictions = np.concatenate([all_predictions, batch_predictions])
    print(batch_predictions)
    print(result)
    print("batch_predictions",batch_predictions)
    print("all_predictions",all_predictions )



  print(time.time() - start)

  if y_test is not None:
    y_test = np.argmax(y_test, axis=1)
    correct_predictions = sum(all_predictions == y_test)
    print(correct_predictions)
    # Save the actual labels back to file

    for idx, example in enumerate(test_examples):
      print(idx, '\t', example)
    # example['tag'] = actual_labels[idx]

    with open('./data/small_samples_prediction.json', 'w') as outfile:
      json.dump(test_examples.to_json(), outfile, indent=4)

    logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
    logging.critical('The prediction is complete')





if __name__ == '__main__':
  tf.app.run()
