import cv2 as cv
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == 1:
        print('B: ', param[y][x][0], '\nG: ', param[y][x][1], '\nR: ', param[y][x][2])
        print('=================================')


Path = 'Data/'
Name = 'rabong2.jpg'
FullName = Path + Name

# 이미지 읽기
img = cv.imread(FullName)

#
#
# 여기에 소스코드 작성
#
#
# 이미지 출력



for y in range(img.shape[0]):
    for x in range(img.shape[1]):
#         ******* 3) 두 이미지의 좌측 상단 모서리가 닿도록 이미지 합치기 ***********:
        if img[y][x][1] > 100 and img[y][x][2] > 100 :
            img[y][x][0] = 0
            img[y][x][1] = 0
            img[y][x][2] = 255
color = [0,0,0,0]

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if y >= 0 and y<=256 and x>=0 and x <= 256 and img[y][x][2] == 255 :
            color[1] = color[1] + 1
        elif y >= 0 and y <= 256 and x>=256 and img[y][x][2] == 255 :
            color[0] = color[0] + 1
        elif y >= 256 and x<=256 and img[y][x][2] == 255 :
            color[2] = color[2] + 1
        elif y >= 256 and x>=256 and img[y][x][2] == 255 :
            color[3] = color[3] + 1
max = 0
max_idx = 0
idx = 0
for i in color:
    if max < i:
        max = i
        max_idx = idx
    idx = idx + 1
print(max_idx + 1)
cv.imshow('img', img)
#cv.imshow('gray1', gray1)
#cv.imshow('gray', gray)
#cv.imshow('blur', blur)


while cv.waitKey(33) <= 0:
    cv.setMouseCallback('img', mouse_callback, img)

cv.waitKey(0)
