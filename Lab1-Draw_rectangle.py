import cv2
import numpy as np
import math as m
drawing = False  
rectangle = None  
background = None 

def create_white_background(width, height):
    #tao mang 3 chieu
    background = np.zeros((height, width, 3))
    background.fill(255)
    return background

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle,xg,yg

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a copy of the background image to draw the rectangle on
            img = background.copy()
            
            # Update the current rectangle coordinates
            xg=x
            yg=y
            # Cac parameter tuong tu la diem dau,diem cuoi,mau,do day
            cv2.rectangle(img, (ix, iy), (xg, yg), (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow('image', img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        # Draw the final rectangle on the background image
        cv2.rectangle(background, (ix, iy), (xg, yg), (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('image', background)

def translate_rectangle(tx, ty):
    # Add giá trị tx,ty để dịch chuyển hình 
    global ix, iy, xg, yg
    ix=ix+tx
    iy=iy+ty
    xg=xg+tx
    yg=yg+ty
    translated_rectangle = (
        ix, iy, xg, yg
    )
    return translated_rectangle

def rotation(angle):
    a = np.array((ix, iy))
    b = np.array((xg, iy))
    c = np.array((xg, yg))
    d = np.array((ix, yg))
    r = np.array([a, b, c, d])

    center = np.array([int((ix + xg) / 2), int((iy+yg) / 2)])
    
    # quy về gốc tọa độ O
    R = r - center

    # xoay theo chieu kim dong ho
    Q = np.array(((m.cos(angle), -m.sin(angle)),
                 (m.sin(angle), m.cos(angle))))
    
    for i in range(R.shape[0]):
        # Nhân ma trận rotation với tâm R
        # Reshape(row,column)
        R[i] = (Q @ R[i].reshape(2,1)).reshape(1,2)
    
    # Trả lại ví trị của các điểm ban đầu
    R = center + R 
    # vẽ hình lên bức ảnh mà các tọa độ pixel nguyên=> round rồi int
    R = R.round()
    R = R.astype(int)
    
    # cv2.polylines(background,R,True,(0,255,255))
    cv2.line(background, R[0], R[1], (0, 0, 255), 2)
    cv2.line(background, R[1], R[2], (0, 0, 255), 2)
    cv2.line(background, R[2], R[3], (0, 0, 255), 2)
    cv2.line(background, R[3], R[0], (0, 0, 255), 2)

def scale_rectangle(scale):
    global ix, iy, xg, yg

    cx = (ix + xg) // 2
    cy = (iy+yg) // 2

    scaled_width = int(abs(xg - ix) * scale)
    scaled_height = int(abs(yg - iy) * scale)
 
    scaled_rectangle = (
        cx - scaled_width // 2, cy - scaled_height // 2,
        cx + scaled_width // 2, cy + scaled_height // 2
    )

    return scaled_rectangle


background = create_white_background(800, 800)

# Ten cua so
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', background)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('t') :
        tx = int(input("Enter the translation along the x-axis: "))
        ty = int(input("Enter the translation along the y-axis: "))
        translated_rectangle = translate_rectangle(tx, ty)
        background=background = create_white_background(800, 800)
        cv2.rectangle(background, (translated_rectangle[0], translated_rectangle[1]),
                      (translated_rectangle[2], translated_rectangle[3]), (255, 0, 0), 2)

    elif key == ord('r') :
        angle = int(input("Enter the rotation angle: "))
        center = (background.shape[1] // 2, background.shape[0] // 2)
        background=background = create_white_background(800, 800)
        rotated_rectangle = rotation(angle)
        
    elif key == ord('s'):
        scale = float(input("Enter the scaling size: "))
        scaled_rectangle = scale_rectangle(scale)
        background=background = create_white_background(800, 800)
        cv2.rectangle(background, (scaled_rectangle[0], scaled_rectangle[1]),
                      (scaled_rectangle[2], scaled_rectangle[3]), (0, 0, 255), 2)

cv2.destroyAllWindows()
