from classes.extract_digits import extract_digits

#for windows
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Ex=extract_digits()

a,b=Ex.return_unit_and_mesure('dataset/testdetection/Echelle/hello.png')

print(str(a)+b)
# print(b)