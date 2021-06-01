import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from imageio import imread
import numpy as np
import tensorflow as tf

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

from PIL import Image
import re, time, base64

from random import randint

app = Flask(__name__)

# Adding Cross Origin Resource Sharing to allow requests made from the front-end
# to be successful.
CORS(app)

# Defining the model configuration files.
# Change these files to add your own model!
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DETECT_PATH = dir_path + '/grabandgo-detection/training/ckpt-9'
PIPELINE_CONFIG_PATH = dir_path + '/grabandgo-detection/pipeline_file.config'
##########################################################
# Menu of loaded groceries, and their respective prices. #
##########################################################


menu = {'item' : 
    { '1':'indomie',
    '2' : 'aqua',
    '3': 'tissue',
    '4': 'chitato',
    '5': 'shampoo',
    '6': 'pepsodent',
    },
    'price' : { '1':2500,
    '2': 12000,
    '3': 5000,
    '4':5000,
    '5':10000,
    '6':12000 }
    }

##################################################
# Utilities
##################################################

def load_image_into_numpy_array(img_data):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = img_data
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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

def getI420FromBase64(codec):
    """ Convert image from a base64 bytes stream to an image. """
    base64_data = re.sub(b'^data:image/.+;base64,', b'', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def readLabels():
    # Read each line of label file and strip \n
    labels = [label.rstrip('\n') for label in open('/model/saved_model/labels.txt')]
    return labels


def apiResponseCreator_det(inputs, outputs):
	return dict(list(zip(inputs,outputs)))


def apiResponseCreator(labels, classifications):
    return dict(zip(labels, classifications))


def printTensors(model_file):
    # read protobuf into graph_def
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for operation in graph.get_operations():
        print(operation.name)


##################################################
# REST API Endpoints For Web App
##################################################


@app.route('/')
def homepage():
    return 'This backend serves as a REST API for the React front end. Try running npm start from the self-checkout folder.'


@app.route('/detection', methods=['POST'])
def detection():
    request.get_data()
    
    # Load in an image to object detect and preprocess it
    img_data = getI420FromBase64(request.data)
    image_np = load_image_into_numpy_array(img_data)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    
    label_id_offset = 1
    # detections = detect_fn(x_input)
    # Setting initial detection time, so execution time can be calculated.    
    t_det = time.time()

    # Get the predictions (output of the softmax) for this image
    #tf_results_det = sess_det.run([output_tensor_det,detection_boxes,detection_scores,detection_num], {input_tensor_det : x_input})

    dt_det = time.time() - t_det
    app.logger.info("Execution time: %0.2f" % (dt_det * 1000.))

    # Different results arrays
    # print(detections)
    predictions_det = detections['detection_classes']
    prediction_scores_det=detections['detection_scores'][0].numpy(),
    prediction_boxes_det= detections['detection_boxes'][0].numpy(),
    prediction_num_det=detections['num_detections'][0]
    print("----------------------------")
    print(predictions_det)
    print(prediction_scores_det)
    print("----------------------------")

    threshold = 0.85

    num=int(prediction_num_det)
    predict_list=(predictions_det[0].numpy() + label_id_offset).astype(int)
    scores = prediction_scores_det[0]
    label=[]

    for i in range(num):
        new_item = {}
        if scores[i] > threshold:
            prediction_label = str(predict_list[i])
            obj_name = menu['item'][prediction_label]
            obj_price = menu['price'][prediction_label]

            new_item = {'id': randint(0, 100000),
                        'name': obj_name,
                        'quantity': 1,
                        'price': obj_price}

            label.append(new_item)
           
    print("number and list of items that above the threshold")
    print(len(label))
    print(label)
    return jsonify(label)


##################################################
# Starting the server
##################################################


if __name__ == '__main__':
    print('Starting TensorFlow Server')

    print('Configuring TensorFlow Graph..')
    # Create the session that we'll use to execute the model
    # sess_config = tf.ConfigProto(
    #     log_device_placement=False,
    #     allow_soft_placement = True
    # )

    print('Loading Model...')
    # tf.keras.backend.clear_session()
    # detect_fn = tf.saved_model.load(MODEL_DETECT_PATH)
    
    # model_dir = str(filenames[-1]).replace('.index','')
    configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
    model_config = configs['model']
    detection_model = model_builder.build(
    model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
    ckpt.restore(MODEL_DETECT_PATH)

    detect_fn = get_model_detection_function(detection_model)

    app.run(debug=False, host='0.0.0.0')
