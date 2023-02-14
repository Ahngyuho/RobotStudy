import cv2 as cv
import numpy as np

Path = ''
Name = 'two.png'
FullName = Path+Name

img = cv.imread(FullName)

# ************ Sobel Filter *******************
# # Sobel X축 Filter
# kernel1 = np.array()
# dst1 = cv.filter2D(img, -1, kernel1)
#
# # Sobel Y축 Filter
# kernel2 = np.array()
# dst2 = cv.filter2D(img, -1, kernel2)
#
# cv.imshow('sobelX', dst1)
# cv.imshow('sobelY', dst2)
# cv.imshow('sum', dst1 + dst2)


# *************** 대각선 방향 edge 검출 필터 *************
# # / 방향 엣지 검출 필터
# kernel3 = np.array()
# dst3 = cv.filter2D(img, -1, kernel3)
#
# # \ 방향 엣지 검출 필터
# kernel4 = np.array()
# dst4 = cv.filter2D(img, -1, kernel4)
#
# cv.imshow('cross1', dst3)
# cv.imshow('cross2', dst4)


# ********* 네 개의 Sobel Filter를 사용하여 모든 방향 Edge 검출하기 ***********
# kernel1 = np.array()
# dst1 = cv.filter2D(img, -1, kernel1)
#
# kernel2 = np.array()
# dst2 = cv.filter2D(img, -1, kernel2)
#
# kernel3 = np.array()
# dst3 = cv.filter2D(img, -1, kernel3)
#
# kernel4 = np.array()
# dst4 = cv.filter2D(img, -1, kernel4)
#
# cv.imshow('sumsum', dst1 + dst2 + dst3 + dst4)


cv.imshow('Original', img)
cv.waitKey(0)
cv.destroyAllWindows()