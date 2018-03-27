import cv2
import numpy as np

# 读取一张斯里兰卡拍摄的大象照片
img = cv2.imread('img/faraway.jpg')

# 仿射变换不同于gamma变换（位置不变，像素点的值变化），仿射变换是对像素点的坐标进行操
# 仿射变换更改了像素点的坐标
# 沿着横纵轴放大1.6倍，然后平移(-160,-140)，最后沿原图大小截取，等效于裁剪并放大
M_crop_faraway = np.array([
    [1.6, 0, -160],
    [0, 1.6, -140]
], dtype=np.float32)

img_faraway = cv2.warpAffine(img, M_crop_faraway, (400, 600))
cv2.imwrite('img/faraway_zoomed.jpg', img_faraway)

# x轴的剪切变换，角度15°
theta = 15 * np.pi / 180
M_shear = np.array([
    [1, np.tan(theta), 0],
    [0, 1, 0]
], dtype=np.float32)

img_sheared = cv2.warpAffine(img, M_shear, (400, 600))
cv2.imwrite('img/faraway_sheared.jpg', img_sheared)

# 顺时针旋转，角度15°
M_rotate = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0]
], dtype=np.float32)

img_rotated = cv2.warpAffine(img, M_rotate, (400, 600))
cv2.imwrite('img/faraway_rotated.jpg', img_rotated)

# 某种变换，具体旋转+缩放+旋转组合可以通过SVD分解理解
M = np.array([
    [1, 1.5, -400],
    [0.5, 2, -100]
], dtype=np.float32)

img_transformed = cv2.warpAffine(img, M, (400, 600))
cv2.imwrite('img/faraway_transformed.jpg', img_transformed)