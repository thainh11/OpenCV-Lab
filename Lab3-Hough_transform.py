import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

def harris_corner_detector(image, block_size=2, ksize=3, k=0.01, threshold=0.01):
    # k: độ nhạy của thuật toán đối với góc
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, block_size, ksize, k)
    corners = cv2.dilate(corners, None)
    image[corners > threshold * corners.max()] = [0, 0, 255] 
    
    return image

def hog_descriptor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_descriptor, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
		visualize=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,10))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Image')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Visualization')
    plt.show()

def canny_operator(image, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    return edges

def hough_transform(image, threshold=100, min_line_length=60, max_line_gap=10):
    edges = canny_operator(image)
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #Ve line mau xanh
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        
    return image

image = cv2.imread('D:\CPV301\images\lane.jpg')

image1=image.copy()

corner_image = harris_corner_detector(image)
hog_features = hog_descriptor(image)
edges = canny_operator(image)
line_image = hough_transform(image1)

# cv2.imshow("Harris Corner Detector", corner_image)
# cv2.moveWindow("Harris Corner Detector", 100, 100)
# cv2.waitKey(0)
# cv2.imshow("Canny Operator", edges)
# cv2.moveWindow("Canny Operator", 100, 100)
# cv2.waitKey(0)
cv2.imshow("Hough Transform", line_image)
cv2.moveWindow("Hough Transform", 100, 100)
cv2.waitKey(0)


