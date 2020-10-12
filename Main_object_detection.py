from classes.object_detection_model import ObjectDetectionModel
import cv2
import tensorflow as tf
print(tf.__version__)
#Initialisation de la classe

#Paramètre soit "detection_echelle" pour détecter les echelles "soit detection_plan_tableau"
#comme l'exemple ci dessus

#
# Obj=object_detection_model("detection_echelle")
Obj=ObjectDetectionModel()
img=cv2.imread("dataset/testdetection/Tableau/2.png")
# ok=cv2.imread("dataset/graphe/95.png")
# ok=cv2.imread("dataset/test_class/1.png")
#
# xmin, xmax, ymax, ymin, THRESHOLD=Obj.detect_scale(img,99)
# print(xmin,xmax,ymax,ymin,THRESHOLD)

#
# Echelledict,THRESHOLD=Obj.detect_scale(img,98)
# print("scale xmin, xmax, ymax, ymin")
# print(Echelledict,THRESHOLD)
#


##
# xmin, xmax, ymax, ymin, THRESHOLD=Obj.detect_echelle(img,99)
# print(xmin,xmax,ymax,ymin,THRESHOLD)
# #
# # Echelledict,THRESHOLD=Obj.detect_echelle(img,98)
# # print("echelle xmin, xmax, ymax, ymin")
# # print(Echelledict,THRESHOLD)
#
# Tabledict, THRESHOLDtable=Obj.detect_table(img,0.025)
Tabledict, THRESHOLDtable=Obj.detect_graphe(img,96)
print("graphe xmin, xmax, ymax, ymin")
print(Tabledict,THRESHOLDtable)


#
# Plandict, THRESHOLDplan=Obj.detect_plan(img,98)
# print("plans xmin, xmax, ymax, ymin")
# print(Plandict,THRESHOLDplan)


#
#
# #Chargement du model et initialisation du model qui se trouve dans le path "dataset/model/detection_echelle"
# sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections=Obj.initialise_model()
#
#
# ##Predire des images dans un dossier spécifique :
# #1er parametre : path contenant les images a prédire .
# #2eme parametre path output dans lequel on veut les images avec le "bounding box"
# #les autres parametres sont les paramétres du graphe du modèle
# dict=Obj.multipleimg_to_predicted_images("dataset/test_class","dataset/testdetection",sess,image_tensor,detection_boxes,detection_scores,detection_classes,num_detections)
# print(dict)
#
