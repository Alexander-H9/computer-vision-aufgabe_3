import numpy as np
import cv2 as cv
from pyzbar.pyzbar import decode

img = cv.imread("b1.jpg")
# img = cv.imread("b2.jpg")
# img = cv.imread("b3.jpg")
# img = cv.imread("b4.jpg")
im = img.copy()

kernel = np.ones((3, 3),np.uint8)

def filter(method, kernel, im):
    print("dilate")
    if method == "dilate":
        for _ in range(3):
            dilation = cv.dilate(im, kernel, iterations = 9)
            im = dilation.copy()

            erosion = cv.erode(im, kernel, iterations = 3)
            im = erosion.copy()

    if method == "erode":
        print("erode")
        for _ in range(4):
            erosion = cv.erode(im, kernel, iterations = 5)
            im = erosion.copy()

            dilation = cv.dilate(im, kernel, iterations = 8)
            im = dilation.copy()

        erosion = cv.erode(im, kernel, iterations = 12) # ausgleichen der differenz zweischen erode und dilate, damit die bildgröße gleichbleibt
        im = erosion.copy()
    return im

im = filter("erode", kernel, im)

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # https://docs.opencv.org/4.0.1/d4/d73/tutorial_py_contours_begin.html
cv.drawContours(im, contours, -1, (0,255,0), 4) # detection of the shape

k = 0
maxValues = 0
index = 0
for i in contours:  # find the index with most values inside (= the contours of the barcode)
    if maxValues < i.size:  # size are the amount of values
        maxValues = i.size
        index = k
    k += 1

print("index ", index , " groeßter wert: ", maxValues)

cnt = contours[index]
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
# im = cv.drawContours(im,[box],0,(0,0,255),8) # detection of the rect around the shape
cv.drawContours(img,[box],0,(0,0,255),4)

# print(box)

pts1 = np.float32(box)
pts2 = np.float32([[0,300],[0,0],[510,0],[510,300]]) # verhältnis 1.7 : 1.  point order in box: ul(0,y), ol(0,0), or(x,0), ur(x,y)
M = cv.getPerspectiveTransform(pts1,pts2)
warpedImg = cv.warpPerspective(img,M,(510,300))

barcode = decode(warpedImg) # decode the transformed img

# print(barcode[0].data)

blank = np.zeros((100,510,3), np.uint8) + 255 # make a blank where the barcode data can be showed

warpedImg = cv.vconcat([warpedImg, blank]) # add the blank vertically to the transformed barcode

if barcode:
    cv.putText(warpedImg, str(barcode[0].data), (20,360), cv.FONT_HERSHEY_COMPLEX, 1, (10, 255, 40), 2) # put the text on it
else:
    print("detection failed")

while True:

    cv.imshow("Barcode", img)
    cv.imshow("Contours", im)
    cv.imshow("Inverted Thresh", thresh)
    cv.imshow("Transformed", warpedImg)

    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):
        break