import os
import cv2
import pytesseract
import re
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

paused = False
cap = None
stop_video = False
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def open_file():
    global cap, stop_video, paused
    stop_video = True
    paused = False

    if cap is not None and cap.isOpened():
        cap.release()
        hien_thi_video.configure(image='')
        hien_thi_video.image = None

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            label_text.config(text="Error: không thể mở ảnh")
            return
        
         # Lấy tên tệp hình ảnh không có phần mở rộng
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

        # Reduce noise using Gaussian blur
        blur = cv2.GaussianBlur(contrast, (5, 5), 0)

        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_plates = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 5000:  # Contour area threshold
                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = thresh[y:y + h, x:x + w]

                    # Read characters from each part of the license plate
                    text = pytesseract.image_to_string(roi, config='--psm 6')
                    text = re.sub(r'[^a-zA-Z0-9-]', '', text)

                    if text and len(text) == 9 and text[2].isalpha() and all(c.isdigit() for i, c in enumerate(text) if i != 2 and i != 3):
                        detected_plates.append((x, y, w, h, text))

        # Filter overlapping or near rectangles
        final_plates = []
        for plate in detected_plates:
            x, y, w, h, text = plate
            if not any((abs(x - fx) < 50 and abs(y - fy) < 50) for fx, fy, fw, fh, ftext in final_plates):
                final_plates.append(plate)

        for plate in final_plates:
            x, y, w, h, text = plate
            # Draw rectangle around the license plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            result_text = "Result is correct" if text == file_name else "Result is incorrect"
            # Display license number on the image
            cv2.putText(img, text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if result_text == "Result is correct" else (0, 0, 255), 2)
            cv2.putText(img, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if result_text == "Result is correct" else (0, 0, 255), 2)

        # Convert image from OpenCV to format suitable for Tkinter
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Display image on the image label
        hien_thi_anh.configure(image=img)
        hien_thi_anh.image = img

def open_video():
    global cap, stop_video, paused
    stop_video = True
    paused = False

    if cap is not None and cap.isOpened():
        cap.release()
        hien_thi_video.configure(image='')
        hien_thi_video.image = None

    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.wmv;*.avi")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        stop_video = False
        if not cap.isOpened():
            label_text.config(text="Error: không thể mở video.")
            return

        # Clear image display
        hien_thi_anh.configure(image='')
        hien_thi_anh.image = None

        def process_frame():
            global cap, stop_video

            if stop_video:
                if cap.isOpened():
                    cap.release()
                return

            if paused:
                root.after(30, process_frame)
                return
            
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                cap.release()
                return
            
              # Lấy tên tệp video không có phần mở rộng
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Increase contrast
            contrast_frame = cv2.convertScaleAbs(gray_frame, alpha=1.5, beta=0)
            blur_frame = cv2.GaussianBlur(contrast_frame, (5, 5), 0)
            _, thresh_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            cnts, _ = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            detected_plates = []

            for c in cnts:
                area = cv2.contourArea(c)
                if area > 5000:
                    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(c)
                        roi = thresh_frame[y:y + h, x:x + w]
                        
                        text = pytesseract.image_to_string(roi, config='--psm 6')
                        text = re.sub(r'[^a-zA-Z0-9-]', '', text)
                        
                        if text and len(text) == 9 and text[2].isalpha() and all(c.isdigit() for i, c in enumerate(text) if i != 2 and i != 3):
                            detected_plates.append((x, y, w, h, text))

            # Filter overlapping or near rectangles
            final_plates = []
            for plate in detected_plates:
                x, y, w, h, text = plate
                if not any((abs(x - fx) < 50 and abs(y - fy) < 50) for fx, fy, fw, fh, ftext in final_plates):
                    final_plates.append(plate)

            for plate in final_plates:
                x, y, w, h, text = plate
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                result_text = "Result is correct" if text == file_name else "Result is incorrect"

                # Display license number and result on the frame
                cv2.putText(frame, text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if result_text == "Result is correct" else (0, 0, 255), 2)
                cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if result_text == "Result is correct" else (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)

            # Display frame on the video label
            hien_thi_video.configure(image=frame)
            hien_thi_video.image = frame
            
            root.after(30, process_frame)

        process_frame()

def toggle_pause(event=None):
    global paused
    paused = not paused         

root = tk.Tk()
root.title("Read License Plate Processing")

menu_bar = tk.Menu(root)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open Image", command=open_file)
file_menu.add_command(label="Open Video", command=open_video)

menu_bar.add_cascade(label="Open File", menu=file_menu)

root.config(menu=menu_bar)

# Create frames to hold the image and video
khung_anh = tk.Frame(root)
khung_anh.pack(padx=5)

# Labels for image and video
nhan_anh = tk.Label(khung_anh)
nhan_anh.pack()

hien_thi_anh = tk.Label(khung_anh)
hien_thi_anh.pack()

nhan_video = tk.Label(khung_anh)
nhan_video.pack()

hien_thi_video = tk.Label(khung_anh)
hien_thi_video.pack()

label_text = tk.Label(root, font=('Helvetica', 16))
label_text.pack()
 
# Bind the space key to stop video processing
root.bind('<space>', toggle_pause)

root.mainloop()
