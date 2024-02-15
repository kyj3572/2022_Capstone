from PIL import Image
import pytesseract
import cv2
import re

#실행파일 위치지정
def ocr():
    pytesseract.pytesseract.tesseract_cmd = "C:/Tesseract-OCR/tesseract.exe"
    image=Image.open("address.jpg")
    
    text1 = pytesseract.image_to_string(image)
    print('all_text :', text1)
    print("\n")

    num = re.findall("\d+", text1)
    print('num_only : ', num)
    print("\n")

    result = ' '.join(s for s in num)
    print('str_num :', result)


    with open('ocr_result.txt','w',encoding='utf8')as f:
        
        f.write(result)
        #f.write("\n\n")
        #f.write(result.replace(" ","")) 

    #fin