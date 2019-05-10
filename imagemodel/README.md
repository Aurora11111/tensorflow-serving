#  install tensorflow serving enviroment

    1. pip install grpcio
       pip install --upgrade tensorflow-serving-api-python3

    2. bazel build //tensorflow_serving/model_servers:tensorflow_model_server

       bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=saved_model --model_base_path=YOUR_MODEL_PATH

# config export_model.py
    1. Line 9. faster_rcnn_inception_v2_pets.config dir in phishing_object_training project
    2. Line 13. training model where phishing_object_training project finally produce
    3. Line 27. your serving model output dir

# run
    python cilent_test
