import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from imageio import imread
import numpy as np
import tensorflow as tf

import numpy as np
from six import BytesIO
from PIL import Image
from models import setup_model
import tensorflow as tf
from PIL import Image
import re, time, base64

from random import randint

app = Flask(__name__)
CORS(app)

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
    return 'REST API Core for Grab and Go Detection Service'


@app.route('/detection', methods=['POST'])
def detection():
    request.get_data()
    # Load in an image to object detect and preprocess it
    img_data = getI420FromBase64(request.data)
    image_np = load_image_into_numpy_array(img_data)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    
    label_id_offset = 1 
    t_det = time.time()

    dt_det = time.time() - t_det
    app.logger.info("Execution time: %0.2f" % (dt_det * 1000.))
    # Different results arrays
    predictions_det = detections['detection_classes']
    prediction_scores_det=detections['detection_scores'][0].numpy(),
    prediction_boxes_det= detections['detection_boxes'][0].numpy(),
    prediction_num_det=detections['num_detections'][0]

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
    return jsonify(label)


##################################################
# Starting the server
##################################################


if __name__ == '__main__':
    print('Starting TensorFlow Server')

    print('Configuring TensorFlow Graph..')

    print('Loading Model...')
    detect_fn = setup_model() 
    app.run(debug=False, host='0.0.0.0')
