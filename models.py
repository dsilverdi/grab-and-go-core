import tensorflow as tf
import os

from object_detection.utils import config_util
from object_detection.builders import model_builder


dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DETECT_PATH = dir_path + '/grabandgo-detection/training/ckpt-9'
PIPELINE_CONFIG_PATH = dir_path + '/grabandgo-detection/pipeline_file.config'

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

def setup_model():
    configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
    model_config = configs['model']
    detection_model = model_builder.build(
    model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
    ckpt.restore(MODEL_DETECT_PATH)

    detect_fn = get_model_detection_function(detection_model)

    return detect_fn