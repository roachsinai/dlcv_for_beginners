import cv2
color_img = cv2.imread('img/faraway.jpg')
print(color_img.shape)

gray_img = cv2.imread('img/faraway.jpg', cv2.IMREAD_GRAYSCALE)
print(gray_img.shape)
print(type(gray_img))

cv2.imwrite('img/gray_faraway.jpg', gray_img)
reload_grayscale = cv2.imread('img/gray_faraway.jpg')
print(reload_grayscale.shape)

cv2.imwrite('img/faraway_imwrite.jpg', color_img, (cv2.IMWRITE_JPEG_QUALITY, 80))
cv2.imwrite('img/faraway_imwrite.png', color_img, (cv2.IMWRITE_PNG_COMPRESSION, 5))