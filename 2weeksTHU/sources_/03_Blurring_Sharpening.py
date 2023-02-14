import cv2 as cv
import numpy as np

Path = 'Data/'
Name1 = 'lenna.tif'
Name2 = 'salt_pepper2.jpg'

src1 = Path + Name1
src2 = Path + Name2

img = cv.imread(src1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(src2, cv.IMREAD_GRAYSCALE)

# ==================================================
# cv2.filter2D(src, ddepth, kernel)
#   src : 입력 영상
#   ddepth : 출력 영상의 데이터 타입, -1로 지정하면 입력 영상과 같은 데이터 타입으로 생성
#     cv2.CV_8U / cv2.CV_32F / cv2.CV_64F
#   kernel : 필터 행렬
# ==================================================


# ==================== 평균값 필터 ====================
# **** 실습 2) 직접 평균값 필터를 만들어 제공된 영상에 컨볼루션 연산 적용하기 ****
# blurfilter = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
# blurfilter = np.array([[1, 1, 1,1,],[1, 1, 1,1,],[1, 1, 1,1,],[1, 1, 1,1,],[1, 1, 1,1,]])
# blurfilter = blurfilter * 1/25
# blurimg = cv.filter2D(img, -1, blurfilter)
# cv.imshow('blurimg', blurimg)
# cv.waitKey()
# exit()
# cv.imwrite('images/blur.jpg', blurimg)
# ==================================================


# ==================== 미디언 필터 ====================
# cv2.medianBlur(src, ksize)
#   src : 원본 이미지
#   ksize: 필터 크기
# ==================================================
# median_img = cv.medianBlur(img2, 5)
#
# cv.imshow('Salt & Pepper Image', img2)
# cv.imshow('median_img', median_img)
# cv.waitKey()
# exit()
# ==================================================


# ==================== 가우시안 필터 ====================
# **** 실습 3) 직접 가우시안 필터를 만들어 제공된 영상에 컨볼루션 연산 적용하기 ****
# gauss_filter = np.array([[1/16, 2/16, 1/16],[2/16, 4/16, 2/16],[1/16, 2/16, 1/16]])
#np.array([[1,4,6,4,1],[4, 16, 24, 16,4],[6, 24, 36 ,24,6],[4, 16, 24, 16,4],[1,4,6,4,1]] / 256)
# gauss_img = cv.filter2D(img, -1, gauss_filter)
#
# cv.imshow('Original', img)
# cv.imshow('gauss_img', gauss_img)
#
# cv.waitKey()
# exit()
# ==================================================


# =============== 가우시안 필터 (함수 사용) ===============
# cv2.getGaussianKernel(ksize, sigma, ktype)
#   ksize : 필터 크기
#   sigma : 가우시안 함수 식 중 시그마 값
#   ktype : 필터 자료형 (default : float64)
# ==================================================
# gauss_filter = cv.getGaussianKernel(5, 3)
# gauss_img = cv.filter2D(img, -1, gauss_filter)
#
# cv.imshow('Original', img)
# cv.imshow('gauss_img', gauss_img)
#
# cv.waitKey()
# exit()
# ==================================================


# =============== 가우시안 필터 (함수 사용 2) ===============
# cv2.GaussianBlur(src, ksize, sigmaX)
#   src : 원본 이미지
#   ksize : 필터 크기, (0, 0)을 할당할 경우 sigmaX 값에 따라 자동으로 설정
#   sigmaX : 가우시안 함수 식 중 시그마값
# ==================================================
# cv.imshow('Original', img)
#
# for sigma in range(1, 5):
#     gauss_img = cv.GaussianBlur(img, (0, 0), sigma)
#     cv.imshow(f'gauss_img{sigma}', gauss_img)
#
# cv.waitKey()
# exit()
# ==================================================



# ==================== 샤프닝 ====================
# **** 실습 4) 직접 샤프닝 필터를 만들어 제공된 영상에 컨볼루션 연산 적용하기 ****
# sharpfilter1 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# sharpfilter2 = np.array([[0,-2,0],[-2,5,-2],[0,-2,0]])
# sharpfilter3 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
# sharp_img1 = cv.filter2D(img, -1, sharpfilter1)
# sharp_img2 = cv.filter2D(img, -1, sharpfilter2)
# sharp_img3 = cv.filter2D(img, -1, sharpfilter3)
#
# cv.imshow('Original', img)
# cv.imshow('sharpimg1', sharp_img1)
# cv.imshow('sharpimg2', sharp_img2)
# cv.imshow('sharpimg3', sharp_img3)
#
# cv.waitKey()
#exit()
# ==================================================

# 경계 검출
# 경계 검출을 위해서는 픽셀값이 급변하는 부분을 찾아야 함
# 이를 연속된 픽셀 값에 미분을 하여 찾아낼 수 있음
# 하지만 픽셀은 연속되지 않아 미분 근사값을 사용해야 함
# 픽셀 미분 근사값 => 붙어 있는 픽셀값을 서로 빼주면 됨
# # 로버츠 커널 생성 ---①
# gx_kernel = np.array([[1,0], [0,-1]])
# gy_kernel = np.array([[0, 1],[-1,0]])
#
# # 커널 적용 ---②
# edge_gx = cv.filter2D(img, -1, gx_kernel)
# edge_gy = cv.filter2D(img, -1, gy_kernel)
#
# # 결과 출력
# merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
# cv.imshow('roberts cross', merged)
# cv.waitKey()

# 프리윗 커널 생성
gx_k = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
gy_k = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])

# 프리윗 커널 필터 적용
edge_gx = cv.filter2D(img, -1, gx_k)
edge_gy = cv.filter2D(img, -1, gy_k)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
cv.imshow('prewitt', merged)
cv.waitKey(0)
cv.destroyAllWindows()

# # 소벨 커널을 직접 생성해서 엣지 검출 ---①
# ## 소벨 커널 생성
# gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
# gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
# ## 소벨 필터 적용
# edge_gx = cv.filter2D(img, -1, gx_k)
# edge_gy = cv.filter2D(img, -1, gy_k)
#
# # 소벨 API를 생성해서 엣지 검출
# sobelx = cv.Sobel(img, -1, 1, 0, ksize=3)
# sobely = cv.Sobel(img, -1, 0, 1, ksize=3)
#
# # 결과 출력
# merged1 = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
# merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
# merged = np.vstack((merged1, merged2))
# cv.imshow('sobel', merged)
# cv.waitKey(0)
# cv.destroyAllWindows()
