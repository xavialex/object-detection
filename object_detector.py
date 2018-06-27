from pathlib import Path
import cv2
import time
import numpy as np
import tensorflow as tf
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# Load the configuration variables from 'properties.txt'
try:
    with open('app.properties') as f:
        data = f.readlines()
        for line in data:
            var = line.split('=')
            if var[0] == 'VIDEO_SOURCE': VIDEO_SOURCE = int(var[1])
            elif var[0] == 'UI_PORT': UI_PORT = int(var[1])
            elif var[0] == 'MODEL': MODEL = var[1].rstrip()
            elif var[0] == 'DISCRIMINATIVE_CLASSES': 
                DISCRIMINATIVE_CLASSES = [int(i) for i in var[1].split(',')]
except:
    print("Error parsing configuration variables from \'properties.txt\'. Using default configuration instead")
    VIDEO_SOURCE = 0
    UI_PORT = 8082
    MODEL = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    DISCRIMINATIVE_CLASSES = [1] # Id's taken from mscoco_label_map.pbtxt
    
# Initial configuration
CWD_PATH = Path('.')
TF_MODELS_PATH = Path('TF Object Detection Models/trained_models')
# Path to frozen detection graph. This is the actual model that is used for the object detection
MODEL_NAME = MODEL #'ssdlite_mobilenet_v2_coco_2018_05_09' 
PATH_TO_CKPT = TF_MODELS_PATH / MODEL_NAME / 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = CWD_PATH / 'object_detection' / 'data' / 'mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(str(PATH_TO_LABELS))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading COCO Id's/classes dictionary
coco_classes_dict = np.load('coco_class_dict.npy').item()

class ObjectDetection:
    """Static variables accesible for the main and the rest of the threads.
    
    Args:
        count (int): Total number of instances detected.
        img (nparray): Encoded image with the results of the prediction.
        data_dict (dict): Info used in 'message'.
            class (str): Class of the instance detected.
            msg (str): Informative message about the object.
            confidence (str): Detected object's confidence.
        message (str): Info about objects detected.
    
    """
    count = None
    data_dict = {'class': None, 'msg': None, 'confidence': None}
    message = bytes('None', 'utf-8')
    img = None
    
def model_load_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(PATH_TO_CKPT), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image, sess, graph, class_id=None):
    """Feed forward an image into the object detection model.
    
    Args:
        image (ndarray): Input image in numpy format (OpenCV format).
        sess: TF session.
        graph: Object detection model loaded before.
        class_id (list): Optional. Id's of the classes you want to detect. 
            Refer to mscoco_label_map.pbtxt' to find out more.
        
    Returns:
        output_dict (dict): Contains the info related to the detections.
        
    """
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0].astype(np.float32)
    
    if class_id is not None:
        discrimine_class(class_id, output_dict)
        
    ############
    #print("{}, {}, {}".format(output_dict['num_detections'], output_dict['detection_scores'], output_dict['detection_boxes'][0]))
        
    return output_dict

def detection_to_json(output_dict):
    """Translate useful information about the detection to JSON.

    Args:
        output_dict (dict): Output of the model for the current frame.
    Returns:
        Information passed to the static variable for future use.
    
    """
    # Get the number of objects detected with enough confidence
    detections = np.where(output_dict['detection_scores'] > 0.5)[0]
    message = []
    if detections.size > 0:
        for i in detections:
            class_name = coco_classes_dict[str(output_dict['detection_classes'][i])]
            ObjectDetection.data_dict['class'] = class_name
            ObjectDetection.data_dict['msg'] = str("Detected {} in camera field \
                    of view".format(class_name))
            ObjectDetection.data_dict['confidence'] = str(output_dict['detection_scores'][i])
            message.append(json.loads(json.dumps(ObjectDetection.data_dict)))
            ObjectDetection.message = str(message).replace("'", '"')
            
def discrimine_class(class_id, output_dict):
    """Keeps the classes of interest of the frame and ignores the others
    
    Args:
        class_id (int): Id's of the classes you want to detect. Refer to 
            'mscoco_label_map.pbtxt' to find out more.
        output_dict (dict): Output if the model once an image is processed.
        
    Returns:
        output_dict (dict): Modified dictionary which just delivers the
            specified class detections.
            
    """
    total_observations = 0 # Total observations per frame
    for i in range(output_dict['detection_classes'].size):
        if output_dict['detection_classes'][i] in class_id and output_dict['detection_scores'][i]>=0.5:
            # The detection is from the desired category and with enough confidence
            total_observations += 1
        elif output_dict['detection_classes'][i] not in class_id:
            # As this is a not desired detection, the score is artificially lowered
            output_dict['detection_scores'][i] = 0.02
    ObjectDetection.count = total_observations
    #print("######################### " + str(total_observations) + " ########################")
    
def visualize_results(image, output_dict):
    """Returns the resulting image after being passed to the model.
    
    Args:
        image (ndarray): Original image given to the model.
        output_dict (dict): Dictionary with all the information provided by the model.
    
    Returns:
        image (ndarray): Visualization of the results form above.
        
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
    
    return image


# Serving a web interface
class HTTPhandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
    def _set_image_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        # Sends data of interest to the frontend
        if 'img' in self.path:
            self._set_image_headers()
            content = ObjectDetection.img
            self.wfile.write(content[1].tobytes())
        elif 'alarms' in self.path:
            self._set_headers()
            message = ObjectDetection.message
            self.wfile.write(bytes(str(message), "utf8"))
        elif 'total' in self.path:
            self._set_headers()
            message = ObjectDetection.count
            self.wfile.write(bytes(str(message), "utf8"))
        else:
            # Loading the resulting web and serving it again
            self._set_headers()
            html_file = open("ui.html", 'r', encoding='utf-8')
            source_code = html_file.read()
            self.wfile.write(bytes(source_code, "utf8"))
      
    # Overriding log messages    
    def log_message(self, format, *args):
        return
        
class HTTPThread(threading.Thread):
    def __init__(self, name):
        super(HTTPThread, self).__init__()
        self.name = name
        self._stop_event = threading.Event()
        
    def run(self):
        server_address = ('', UI_PORT)
        self.httpd = HTTPServer(server_address, HTTPhandler)
        self.httpd.serve_forever()

    def stop(self):
       self.httpd.shutdown()
       self.stopped = True
       
    def stopped(self):
       return self._stop_event.is_set()

  
def main():
    detection_graph = model_load_into_memory()
    
    # HTTP thread starting in background
    http_thread = HTTPThread("HTTP Publisher Thread")
    http_thread.daemon = True
    http_thread.start()
    
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    
    try:
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:  
                    # Camera detection loop
                    _, frame = video_capture.read()
                    # Change color gammut to feed the frame into the network
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t = time.time()
                    output = run_inference_for_single_image(frame, sess, 
                        detection_graph, DISCRIMINATIVE_CLASSES)
                    processed_image = visualize_results(frame, output)
                    detection_to_json(output)
                    cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                    ObjectDetection.img = cv2.imencode('.jpeg', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                    print('Elapsed time: {:.2f}'.format(time.time() - t))
            
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                     
    except KeyboardInterrupt:   
        pass
    
    print("Ending resources")
    cv2.destroyAllWindows()
    video_capture.release()
    http_thread.stop()


if __name__ == '__main__':
    main()
    