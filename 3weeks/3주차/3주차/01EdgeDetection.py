import cv2 as cv
import numpy as np

Path = ''
Name = 'homework.jpg'
FullName = Path+Name
img = cv.imread(FullName)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img = cv.imread(FullName, cv.IMREAD_GRAYSCALE)


# ********* 기본 1X2, 2X1 필터 *********
# kernel1 = np.array([-1,1])
# dst1 = cv.filter2D(img, -1, kernel1)
#
# # Y축 Filter
# kernel2 = np.array([[-1],[1]])
# dst2 = cv.filter2D(img, -1, kernel2)
#
# cv.imshow('X', dst1)
# cv.imshow('Y', dst2)
# cv.imshow('Sum', dst1 + dst2)
# Solbel Filter
# kernel1 = np.array([[-1,0,1], [-2, 0, 2], [-1, 0, 1]])
# dst1 = cv.filter2D(img, -1, kernel1)
# kernel3 = np.array([[0, 1, 2], [-1, 0, 1], [-3,-1,0]])
# dst3 = cv.filter2D(img, -1, kernel3)
# # Y축 Filter
# kernel2 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
# dst2 = cv.filter2D(img, -1, kernel2)
# kernel4 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
# dst4 = cv.filter2D(img, -1, kernel4)
# # cv.imshow('X', dst1)
# # cv.imshow('Y', dst2)
# cv.imshow('Sum', dst1 + dst2 + dst3 + dst4)

# X축 Filter
# kernel1 = np.array([[-3,-1,0], [-1, 0, 1], [0, 1, 2]])
# dst1 = cv.filter2D(img, -1, kernel1)
# kernel3 = np.array([[0, 1, 2], [-1, 0, 1], [-3,-1,0]])
# dst3 = cv.filter2D(img, -1, kernel3)
# # Y축 Filter
# kernel2 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
# dst2 = cv.filter2D(img, -1, kernel2)
# kernel4 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
# dst4 = cv.filter2D(img, -1, kernel4)
# # cv.imshow('X', dst1)
# # cv.imshow('Y', dst2)
# cv.imshow('Sum', dst1 + dst2 + dst3 + dst4)


# ************* Roberts Cross Filter *********
# # X축 Filter
# roberts_x = np.array()
# RobertsX = cv.filter2D(img, -1, roberts_x)
#
# # Y축 Filter
# roberts_y = np.array()
# RobertsY = cv.filter2D(img, -1, roberts_y)
#
# cv.imshow('robertsX', RobertsX)
# cv.imshow('robertsY', RobertsY)
# cv.imshow('Roberts', RobertsX + RobertsY)


# ************ Prewitt Filter *********
# # X축 Filter
# prewitt_x = np.array()
# PrewittX = cv.filter2D(img, -1, prewitt_x)
#
# # Y축 Filter
# prewitt_y = np.array()
# PrewittY = cv.filter2D(img, -1, prewitt_y)
#
# cv.imshow('prewittX', PrewittX)
# cv.imshow('prewittY', PrewittY)
# cv.imshow('Prewitt', PrewittX + PrewittY)


# ************ Scharr Filter *********
# # X축 Filter
# scharr_x = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
# ScharrX = cv.filter2D(img, -1, scharr_x)
#
# # Y축 Filter
# scharr_y = np.array([[-3,-10,-3],[0,0,0],[3,10,3]])
# ScharrY = cv.filter2D(img, -1, scharr_y)
#
# cv.imshow('scharrX', ScharrX)
# cv.imshow('scharrY', ScharrY)
# cv.imshow('Scharr', ScharrX + ScharrY)


# ************ Laplacian Filter ***********
# X축 Filter
# laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
# Laplacian = cv.filter2D(img, -1, laplacian)
#
# cv.imshow('Laplacian', Laplacian)

sharpfilter = np.array([[1,2,1],[2,4,2],[1,2,1]])
sharpfilter = sharpfilter / 16

sharp_img = cv.filter2D(img, -1, sharpfilter)
img = sharp_img

canny = cv.Canny(img,50,255)
img = canny

cv.imshow('Original', img)
cv.waitKey(0)
cv.destroyAllWindows()