#pip install opencv-python
import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from pytesseract import image_to_string
#import imutils

class extract_digits:
    def __init__(self):
        pass



    def only_numerics(self,seq):
        seq_type= type(seq)
        return seq_type().join(filter(seq_type.isdigit, seq))

    def only_alphanumerics(self,seq):
        seq_type= type(seq)
        return seq_type().join(filter(seq_type.isalpha, seq))


    def extraction_digits(self,imagepath):

        # config='outputbase digits'
        config = ("-l eng --oem 1 --psm 7")

        text = pytesseract.image_to_string(Image.open(imagepath), config=config)
        # text = pytesseract.image_to_string(Image.open("/home/technique/firas/Projects/image_classification/dataset/testdetection/17.png"), config=config)

        x = text.split(" ")
        print(x)
        unitemaxvalue=self.only_numerics(x[len(x) - 1])
        unitemesure=self.only_alphanumerics(x[len(x) - 1])
        return unitemaxvalue,unitemesure



    def return_unit_and_mesure(self,imagepath):
        plt.rcParams['figure.figsize'] = [15, 8]

        # loading image form directory
        img = cv2.imread(imagepath,0)
        #img = cv2.imread('dataset/testdetection/8.png',0)
        # cv2.imshow("hello",img)
        # cv2.waitKey(0)
        # img.shape

        # showing image
        # imgplot = plt.imshow(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
        # plt.show()


        # for adding border to an image
        img1= cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255])

        img123 = img1.copy()

        # Thresholding the image
        (thresh, th3) = cv2.threshold(img1, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)



        # imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
        # plt.show()
        #

        # to flip image pixel values
        th3 = 255-th3

        # imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
        # plt.show()
        #
        # initialize kernels for table boundaries detections
        if(th3.shape[0]<1000):
            ver = np.array([[1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1]])
            hor = np.array([[1,1,1,1,1,1]])

        else:
            ver = np.array([[1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1]])
            hor = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])




        #
        # imgplot = plt.imshow(cv2.resize(verticle_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
        # plt.show()
        #

        # to detect horizontal lines of table borders
        img_hor = cv2.erode(th3, hor, iterations=50)
        hor_lines_img = cv2.dilate(img_hor, hor, iterations=50)
        # imgplot = plt.imshow(hor_lines_img,cmap='gray')

        try:
            contours, hierarchy = cv2.findContours(hor_lines_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rc = cv2.minAreaRect(contours[-1])
            box = cv2.boxPoints(rc)
            xmin=box[0][0]
            xmax=box[2][0]
            print("xmin")
            print(xmin)
            print("xmax")
            print(xmax)
            for p in box:
                pt = (p[0],p[1])
                # print(pt)
                cv2.circle(hor_lines_img,pt,5,(200,0,0),2)


            # cv2.imshow("plank", hor_lines_img)

            unitemaxvalue,unitemesure=self.extraction_digits(imagepath)

            unitemaxvalue=float(unitemaxvalue)
        except:
            xmax=0
            xmin=0
            unitemaxvalue=1
            unitemesure="Could not detect"

        # cv2.waitKey()

        return (xmax-xmin)/unitemaxvalue,unitemesure


