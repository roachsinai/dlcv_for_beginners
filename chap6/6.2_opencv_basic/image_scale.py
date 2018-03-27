import cv2

# 读取一张四川大录古藏寨的照片
img = cv2.imread('img/faraway.jpg')

# 缩放成200x200的方形图像
img_200x200 = cv2.resize(img, (200, 200))

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_200x300 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, 
                              interpolation=cv2.INTER_NEAREST)

# 在上张图片的基础上，上下各贴50像素的黑边，生成400x700的图像
img_400x700 = cv2.copyMakeBorder(img, 50, 50, 0, 0, 
                                       cv2.BORDER_REFLECT, 
                                       value=(0, 0, 0))

# 对照片中树的部分进行剪裁
patch_human = img[150:340, 175:300]

cv2.imwrite('img/cropped_human.jpg', patch_human)
cv2.imwrite('img/resized_200x200.jpg', img_200x200)
cv2.imwrite('img/resized_200x300.jpg', img_200x300)
cv2.imwrite('img/bordered_400x700.jpg', img_400x700)