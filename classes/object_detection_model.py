from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
from utils import label_map_util
# from utils import visualization_utils as vis_util
import sys
import os
import numpy as np
import subprocess


global_IMG_WIDTH = 0
global_IMG_HEIGHT = 0
# model_learnt=None

CWD_PATH =os.getcwd()
loaded_table_plan=0
loaded_echelle=0
loaded_graphe=0
sess=None
image_tensor=None
detection_boxes=None
detection_scores=None
detection_classes=None
num_detections=None

def getprojectpath():

    path = subprocess.Popen("pwd",
                            shell=True,
                            stdout=subprocess.PIPE,
                            universal_newlines=True).communicate()[0]

    path =path.strip('\n')+"/dataset/"
    return path


class ObjectDetectionModel:

    def __init__(self):
        pass

    def initialise_model(self,nc,modelname):
        # create output directory if it doesn't exist
        global CWD_PATH

        global sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

        # This is needed since the notebook is stored in the object_detection folder.
        sys.path.append("..")


        # Name of the directory containing the object detection module we're using
        # MODEL_NAME = 'inference_graph' # The path to the directory where frozen_inference_graph is stored.
        MODEL_NAME = "model/"+modelname+"/"


        # Grab path to current working directory

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'objectdetection.pbtxt')

        # Path to image
        #PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

        # Number of classes the object detector can identify
        NUM_CLASSES = nc
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        global category_index
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')



            sess = tf.Session(graph=detection_graph)


            # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.

        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


        # Number of objects detected

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        return sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections


    def detect_echelle(self, image,THRESHOLD):
            global loaded_echelle
            if(loaded_echelle==0):
                self.initialise_model(2,"detection_echelle")
                loaded_echelle=loaded_echelle+1


            global CWD_PATH,sess, image_tensor, detection_boxes, detection_scores,detection_classes, num_detections

            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            #################### CROP  ##############################""
            THRESHOLD = THRESHOLD/100

            (frame_height, frame_width) = image.shape[:2]


            Echelledict = {}
            for i in range(len(np.squeeze(scores))):

                if np.squeeze(scores)[i] < THRESHOLD:
                    continue


                if (np.squeeze(classes)[i] == 2.0):

                    ymin = int((np.squeeze(boxes)[i][0] * frame_height))
                    xmin = int((np.squeeze(boxes)[i][1] * frame_width))
                    ymax = int((np.squeeze(boxes)[i][2] * frame_height))
                    xmax = int((np.squeeze(boxes)[i][3] * frame_width))
                    # cropped_img = image[ymin:ymax, xmin:xmax]
                    Echelledict[str(i + 1)] = [xmin, xmax, ymax, ymin]

            return Echelledict, THRESHOLD

    def detect_scale(self, image, THRESHOLD):
        global loaded_echelle
        if (loaded_echelle == 0):
            self.initialise_model(2, "detection_echelle")
            loaded_echelle = loaded_echelle + 1

        global CWD_PATH, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #################### CROP  ##############################""
        THRESHOLD = THRESHOLD / 100

        (frame_height, frame_width) = image.shape[:2]

        scaledict = {}
        for i in range(len(np.squeeze(scores))):

            if np.squeeze(scores)[i] < THRESHOLD:
                continue

            if (np.squeeze(classes)[i] == 1.0):
                ymin = int((np.squeeze(boxes)[i][0] * frame_height))
                xmin = int((np.squeeze(boxes)[i][1] * frame_width))
                ymax = int((np.squeeze(boxes)[i][2] * frame_height))
                xmax = int((np.squeeze(boxes)[i][3] * frame_width))
                # cropped_img = image[ymin:ymax, xmin:xmax]
                scaledict[str(i + 1)] = [xmin, xmax, ymax, ymin]
        print(scaledict)
        return scaledict, THRESHOLD

    def detect_table(self, image, THRESHOLD):
        global loaded_echelle
        if (loaded_table_plan == 0):
            self.initialise_model(3, "detection_plan et tableau")
            loaded_echelle = loaded_echelle + 1

        global  CWD_PATH, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #################### CROP  ##############################""
        THRESHOLD = THRESHOLD / 100

        (frame_height, frame_width) = image.shape[:2]
        Tabledict = {}
        for i in range(len(np.squeeze(scores))):

            if np.squeeze(scores)[i] < THRESHOLD:
                continue

            if (np.squeeze(classes)[i] == 2.0):
                ymin = int((np.squeeze(boxes)[i][0] * frame_height))
                xmin = int((np.squeeze(boxes)[i][1] * frame_width))
                ymax = int((np.squeeze(boxes)[i][2] * frame_height))
                xmax = int((np.squeeze(boxes)[i][3] * frame_width))
                cropped_img = image[ymin:ymax, xmin:xmax]
                cv2.imwrite("cropped_img"+str(xmin)+".png",cropped_img)

                print(xmin, xmax, ymax, ymin)

                Tabledict[str(i + 1)] = [xmin, xmax, ymax, ymin]

        return Tabledict, THRESHOLD



    def detect_plan(self, image, THRESHOLD):
        global loaded_echelle
        if (loaded_table_plan == 0):
            self.initialise_model(3, "detection_plan et tableau")
            loaded_echelle = loaded_echelle + 1

        global  CWD_PATH, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #################### CROP  ##############################""
        THRESHOLD = THRESHOLD / 100

        (frame_height, frame_width) = image.shape[:2]
        Plandict = {}
        xmin, xmax, ymax, ymin=0,0,0,0
        for i in range(len(np.squeeze(scores))):

            if np.squeeze(scores)[i] < THRESHOLD:
                continue

            if (np.squeeze(classes)[i] == 1.0):
                ymin = int((np.squeeze(boxes)[i][0] * frame_height))
                xmin = int((np.squeeze(boxes)[i][1] * frame_width))
                ymax = int((np.squeeze(boxes)[i][2] * frame_height))
                xmax = int((np.squeeze(boxes)[i][3] * frame_width))
                # cropped_img = image[ymin:ymax, xmin:xmax]

                Plandict[str(i + 1)] = [xmin, xmax, ymax, ymin]

            return Plandict, THRESHOLD




    def detect_graphe(self, image, THRESHOLD):
        global loaded_graphe
        if (loaded_graphe == 0):
            print("load")
            self.initialise_model(4, "detection_graphe")
            loaded_graphe = loaded_graphe + 1

        global CWD_PATH, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        #################### CROP  ##############################""
        THRESHOLD = THRESHOLD / 100

        (frame_height, frame_width) = image.shape[:2]
        Graphdict = {}
        xmin, xmax, ymax, ymin = 0, 0, 0, 0
        print(len(np.squeeze(scores)))
        for i in range(len(np.squeeze(scores))):

            if np.squeeze(scores)[i] < THRESHOLD:
                continue

            if (np.squeeze(classes)[i] == 4.0):
                ymin = int((np.squeeze(boxes)[i][0] * frame_height))
                xmin = int((np.squeeze(boxes)[i][1] * frame_width))
                ymax = int((np.squeeze(boxes)[i][2] * frame_height))
                xmax = int((np.squeeze(boxes)[i][3] * frame_width))
                # cropped_img = image[ymin:ymax, xmin:xmax]

                Graphdict[str(i + 1)] = [xmin, xmax, ymax, ymin]

        return Graphdict, THRESHOLD









