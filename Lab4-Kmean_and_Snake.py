import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.io import imread
from sklearn.cluster import MeanShift, estimate_bandwidth

img =cv2.imread("D:\CPV301\Code CPV\lenna.jpg")
img1=img.copy()
img2=img.copy()
img3=img.copy()
def snake():
    global img
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3, preserve_range=False),
                        init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()

def watershed_algorithm(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    thresh = threshold_otsu(grayscale_image)
    binary_image = grayscale_image > thresh

    distance = ndi.distance_transform_edt(binary_image)

    markers = ndi.label(binary_image)[0]
    labels = watershed(-distance, markers, mask=binary_image)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(labels)
    ax.axis('off')
    plt.show()
def kmeans_image_segmentation(image, n_clusters):
   
    flattened_image = image.reshape((-1, 3)) 
    flattened_image = np.float32(flattened_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    ret, labels, centers = cv2.kmeans(flattened_image, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image


def mean_shift_segmentation(image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float64) / 255

    image_flattened = image.reshape(-1, 3)
    bandwidth = estimate_bandwidth(image_flattened, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image_flattened)
    labels = ms.labels_
    segmented_image = labels.reshape(image.shape[:2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(segmented_image)
    ax2.set_title('Mean Shift Segmentation')
    plt.show()



snake()
watershed_algorithm(img1)
k_means = kmeans_image_segmentation(img2,7)
mean_shift_segmentation(img3)

cv2.imshow("K-Means", k_means)
cv2.waitKey(0)
cv2.destroyAllWindows()