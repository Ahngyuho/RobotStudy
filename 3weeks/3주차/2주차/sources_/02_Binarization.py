import cv2 as cv

Path = 'Data/'
Name = 'adaptive_thresholdiry.jpg'
# Name = 'lenna.tif'
src = Path + Name

img = cv.imread(src, cv.IMREAD_GRAYSCALE)
cv.imshow('Original Image', img)


# =========== 이진화 (고정 임계값, 함수 사용) ===========
# cv2.threshold(src, thresh, maxval, type)
#   src : 입력 영상
#   thresh : 임계값
#   maxval : 임계값보다 클 때 적용되는 최대 값
#   type : 임계값 적용 방법
#     cv2.THRESH_BINARY : 픽셀 값이 thresh보다 크면 maxval, 작으면 0으로 할당
# ==================================================
bin_img = img.copy()

_, bin_img = cv.threshold(bin_img, 120, 255, cv.THRESH_BINARY)
print(bin_img)

cv.imshow('binarization 3', bin_img)
cv.waitKey()
# ==================================================


# =========== 이진화 (가변 임계값, 함수 사용) ===========
# cv2.adaptiveThreshold(src, maxValue, maxValue, adaptiveMethod, thresholdType, blockSize, C)
#   src : 입력 영상
#   maxValue : 임계값보다 클 때 적용되는 최대 값
#   adaptiveMethod : 적용할 지역 이진화 알고리즘
#     cv2.ADAPTIVE_THRESH_MEAN_C : 주변 영역의 평균값으로 결정
#   thresholdType : 임계값 적용 방법
#     cv2.THRESH_BINARY : 픽셀 값이 thresh보다 크면 maxval, 작으면 0으로 할당
#   blockSize : 지역 이진화를 할 때 고려할 주변 픽셀 크기 (블록 크기)
#   C : 블록 내 평균값에서 뺄 값, 임계 값을 결정하는 파라미터
# ==================================================
# bin_img = img.copy()
#
# bin_img = cv.adaptiveThreshold(bin_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,  15, 2)
# print(bin_img)
#
# cv.imshow('binarization 4', bin_img)
# cv.waitKey()
# ==================================================