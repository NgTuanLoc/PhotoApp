import cv2 as cv

def convertToRGB(image_path="./data/test/img.png"):
    image=cv.imread(image_path)
    cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imwrite("./data/converted/img_converted.png", image)

convertToRGB()