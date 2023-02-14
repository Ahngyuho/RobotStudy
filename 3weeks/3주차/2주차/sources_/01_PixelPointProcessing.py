import cv2 as cv
import numpy as np

Path = 'Data/'
Name = 'lenna.tif'
src = Path + Name

color_img = cv.imread(src)
img = cv.imread(src, cv.IMREAD_GRAYSCALE)
# img = img.astype('float64') / 255
print(img.dtype)


# =============== 산술 연산 1 ===============
# arith_img = img.copy()
#
# for y in range(arith_img.shape[0]):
#     for x in range(arith_img.shape[1]):
#         if arith_img[y][x] + 100 <= 255:
#             arith_img[y][x] = arith_img[y][x] + 100
#         else:
#             arith_img[y][x] = 255
#
# cv.imshow('Original Image', img)
# cv.imshow('Arithmetic Operation 1', arith_img)
#
# cv.waitKey()
# exit()
# ==================================================


# =============== 산술 연산 2 ===============
# arith_img2 = img.copy()
#
# arith_img2 = cv.add(arith_img2, 100)
#
# cv.imshow('Original Image', img)
# cv.imshow('Arithmetic Operation 2', arith_img2)
#
# cv.waitKey()
# exit()
# ==================================================


# =================== 논리 연산 ===================
# img = img / 255
# mask = np.zeros(img.shape, np.float64)
# start_pos = 150
# end_pos = 400
#
# for y in range(start_pos, end_pos):
#     for x in range(start_pos, end_pos):
#         mask[y][x] = 1
#
# # 슬라이싱 기법 사용
# # mask[start_pos:end_pos, start_pos:end_pos] = 1
#
# logic_img = img * mask
#
# cv.imshow('Original Image', img)
# cv.imshow('mask', mask)
# cv.imshow('Logical Operation', logic_img)
#
# cv.waitKey()
# exit()
# ==================================================


# =================== 그레이스케일 ===================
# img = color_img.copy()
# img = img[:, :, 2] / 3 + img[:, :, 1] / 3 + img[:, :, 0] / 3
#
# cv.imshow('Original Image', color_img)
# cv.imshow('grayscale', img.astype('uint8'))
#
# cv.waitKey()
# exit()
# ==================================================


# =============== 전역 이진화 (고정 임계 값) ===============
# # **** 실습 1) for문을 이용하여 threshold 값보다 작은 픽셀 값은 0, 큰 픽셀 값은 255를 할당하여 영상 이진화하기 ****
# binimg = img.copy()
# threshold = 127
#
# for y in range(binimg.shape[0]):
#     for x in range(binimg.shape[1]):
#
# cv.imshow('Original Image', img)
# cv.imshow('binarization 1', binimg)
# cv.waitKey()
# exit()
# ==================================================


# ==== 전역 이진화 (고정 임계 값, 조건문을 이용한 인덱싱) ====
# binimg = img.copy()
# threshold = 127
#
# binimg[binimg <= threshold] = 0
# binimg[binimg > threshold] = 255
#
# cv.imshow('Original Image', img)
# cv.imshow('binarization 2', binimg)
# cv.waitKey()
# exit()
# ==================================================
