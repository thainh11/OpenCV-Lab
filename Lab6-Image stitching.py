import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk

def image_stitching(image1, image2):

    stitcher = cv2.Stitcher_create()

    status, stitched_image = stitcher.stitch([image1, image2])

    if status == cv2.Stitcher_OK:
        stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(stitched_image)
        image.thumbnail((1000, 1000))
        tk_image = ImageTk.PhotoImage(image)
        result_label.configure(image=tk_image)
        result_label.image = tk_image
    else:
        print("Stitching failed!")

def stitch_images():
    root = tk.Tk()
    root.withdraw()
    image1_path = filedialog.askopenfilename(title="Select the first image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))
    image2_path = filedialog.askopenfilename(title="Select the second image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    image_stitching(image1, image2)

window = tk.Tk()
window.title("Image Stitching")
window.geometry("800x600")

stitch_button = tk.Button(window, text="Stitch Images", command=stitch_images)
stitch_button.pack(pady=20)

result_label = tk.Label(window)
result_label.pack()

window.mainloop()
