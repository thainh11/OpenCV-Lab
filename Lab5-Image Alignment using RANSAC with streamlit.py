import streamlit as st
import cv2
import numpy as np

def align_images(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None)
    img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None)
    st.image(np.hstack((img1_with_keypoints, img2_with_keypoints)), channels="RGB", use_column_width=True)
   
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    return aligned_img

st.title("Image Alignment using RANSAC")
st.subheader("Upload Images")
uploaded_files = st.file_uploader("Upload two images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    img1 = cv2.imdecode(np.frombuffer(uploaded_files[0].read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(uploaded_files[1].read(), np.uint8), cv2.IMREAD_COLOR)

    aligned_img = align_images(img1, img2)
    aligned_img_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    st.image(aligned_img_rgb, channels="RGB")

# Using streamlit run Lab5.py to run this file