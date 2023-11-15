import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def color_balance(image, b_ratio, g_ratio, r_ratio):
    # Chia các kênh màu
    b, g, r = cv2.split(image)

    # clip(value, min, max)
    # đảm bảo các kênh màu có giá trị từ 0 đến 255
    b = np.clip(b * b_ratio, 0, 255).astype(np.uint8) #8bit
    g = np.clip(g * g_ratio, 0, 255).astype(np.uint8)
    r = np.clip(r * r_ratio, 0, 255).astype(np.uint8)

    # Gộp các kênh màu đã cân bằng thành bức ảnh mới
    img_balanced = cv2.merge([b, g, r])

    return img_balanced

def update_color_balance(val):
    # Tạo giá trị cho các thanh slide
    b_ratio = slider_b.val
    g_ratio = slider_g.val
    r_ratio = slider_r.val

    # Thực hiện cac bàng màu
    balanced_image = color_balance(image, b_ratio, g_ratio, r_ratio)
    
    ax_image.imshow(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB))
    
    ax_hist.clear()
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist_values = cv2.calcHist([balanced_image], [i], None, [256], [0, 256])
        ax_hist.plot(hist_values, color=color)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_title('Histogram')
    
    #Lien tuc cap nhat histogram
    fig.canvas.draw_idle()

def median_filter(image, kernel_size):
    return cv2.medianBlur(image_fail, kernel_size)

def mean_filter(image, kernel_size):
    return cv2.blur(image_fail, (kernel_size, kernel_size))

def gaussian_smoothing(image, sigma):
    return cv2.GaussianBlur(image_fail, (3, 3), sigma)# sigma=standard deviation

# Load the image
image = cv2.imread('D:\CPV301\Code CPV\lenna.jpg')

image_fail=cv2.imread("D:\CPV301\Code CPV\lenna_fail.jpg")

# Tạo console
fig, (ax_image, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


colors = ('r', 'g', 'b')

for i, color in enumerate(colors):
    hist_values = cv2.calcHist([image], [i], None, [256], [0, 256])
    ax_hist.plot(hist_values, color=color)
ax_hist.set_xlim([0, 255])
ax_hist.set_title('Histogram')


# axes([left, bottom, width, height])
slider_b_ax = plt.axes([0.25, 0.2, 0.65, 0.03])
slider_g_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_r_ax = plt.axes([0.25, 0.1, 0.65, 0.03])


# Create the sliders
#Slider(ax,label,valmin,valmax,valinit=giá trị ban đầu)
slider_b = Slider(slider_b_ax, 'Blue', 0.0, 2.0, valinit=1.0)
slider_g = Slider(slider_g_ax, 'Green', 0.0, 2.0, valinit=1.0)
slider_r = Slider(slider_r_ax, 'Red', 0.0, 2.0, valinit=1.0)


slider_b.on_changed(update_color_balance)
slider_g.on_changed(update_color_balance)
slider_r.on_changed(update_color_balance)

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, wspace=0.2)

plt.show()


# Filter
median_filtered = median_filter(image, kernel_size=3)
mean_filtered = mean_filter(image, kernel_size=3)
gaussian_smoothed = gaussian_smoothing(image, sigma=1.5)



cv2.imshow("Original Image", image_fail)
cv2.moveWindow("Original Image", 100, 100)  

cv2.imshow("Median Filtered Image", median_filtered)
cv2.moveWindow("Median Filtered Image", 400, 100)  

cv2.imshow("Mean Filtered Image", mean_filtered)
cv2.moveWindow("Mean Filtered Image", 700, 100)  

cv2.imshow("Gaussian Smoothed Image", gaussian_smoothed)
cv2.moveWindow("Gaussian Smoothed Image", 1000, 100)  

cv2.waitKey(0)
cv2.destroyAllWindows()