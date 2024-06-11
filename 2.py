import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageHandler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Handler")
        self.master.geometry("1500x700")  # Увеличен для размещения трех изображений

        self.original_img = None
        self.modified_img = None
        self.secondary_img = None
        self.watercolor_img = None

        # Frame for the original and processed images
        self.img_frame = Frame(self.master)
        self.img_frame.pack(side=TOP, pady=10)

        # Original image panel
        self.original_panel = Label(self.img_frame)
        self.original_panel.pack(side=LEFT, padx=10)

        # Processed image panel
        self.modified_panel = Label(self.img_frame)
        self.modified_panel.pack(side=LEFT, padx=10)

        # New image panel
        self.secondary_panel = Label(self.img_frame)
        self.secondary_panel.pack(side=LEFT, padx=10)

        # Frame for the buttons
        self.control_frame = Frame(self.master)
        self.control_frame.pack(side=TOP, pady=10)

        # Frame for the sliders
        self.slider_frame = Frame(self.master)
        self.slider_frame.pack(side=TOP, pady=5)

        self.load_button = Button(self.control_frame, text="Load Image", command=self.load_img)
        self.load_button.pack(side=LEFT, padx=10)

        # Channel extraction menu
        self.channel_var = StringVar(self.master)
        self.channel_var.set("Red")  # Set default value
        self.channel_menu = OptionMenu(self.control_frame, self.channel_var, "Red", "Green", "Blue", command=self.extract_channel)
        self.channel_menu.pack(side=LEFT, padx=10)

        # Buttons for loading and processing image

        self.gray_button = Button(self.control_frame, text="Gray Scale", command=self.to_gray)
        self.gray_button.pack(side=LEFT, padx=10)

        self.sepia_button = Button(self.control_frame, text="Sepia", command=self.to_sepia)
        self.sepia_button.pack(side=LEFT, padx=10)

        self.brightness_contrast_button = Button(self.control_frame, text="Brightness & Contrast", command=self.show_brightness_contrast_sliders)
        self.brightness_contrast_button.pack(side=LEFT, padx=10)

        self.hsv_button = Button(self.control_frame, text="Convert to HSV", command=self.show_hsv_slider)
        self.hsv_button.pack(side=LEFT, padx=10)

        self.median_blur_button = Button(self.control_frame, text="Median Blur", command=self.apply_median_blur)
        self.median_blur_button.pack(side=LEFT, padx=10)

        self.filter_button = Button(self.control_frame, text="Custom Filter", command=self.show_filter_fields)
        self.filter_button.pack(side=LEFT, padx=10)

        self.watercolor_button = Button(self.control_frame, text="Watercolor Filter", command=self.show_watercolor_options)
        self.watercolor_button.pack(side=LEFT, padx=10)

        self.cartoon_button = Button(self.control_frame, text="Cartoon Filter", command=self.show_cartoon_filter_slider)
        self.cartoon_button.pack(side=LEFT, padx=10)

        # Sliders for cartoon filter threshold, brightness and contrast (initially hidden)
        self.threshold_slider = Scale(self.slider_frame, from_=1, to=50, orient=HORIZONTAL, label="Cartoon Threshold")
        self.threshold_slider.set(100)

        self.brightness_slider = Scale(self.slider_frame, from_=-100, to=100, orient=HORIZONTAL, label="Contrast")
        self.brightness_slider.set(0)

        self.contrast_slider = Scale(self.slider_frame, from_=1, to=3, orient=HORIZONTAL, label="Brightness", resolution=0.1)
        self.contrast_slider.set(1)

        # Slider for HSV hue adjustment (initially hidden)
        self.hue_slider = Scale(self.slider_frame, from_=0, to=180, orient=HORIZONTAL, label="Hue")
        self.hue_slider.set(0)

        # Slider for Watercolor mix
        self.watercolor_slider = Scale(self.slider_frame, from_=0, to=100, orient=HORIZONTAL, label="Watercolor Mix")
        self.watercolor_slider.set(50)

        self.filter_matrix_entries = []
        self.apply_filter_button = None
        self.add_image_button = None

    def load_img(self):
        self.hide_sliders()
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_img = cv2.imread(file_path)
            self.modified_img = self.original_img.copy()
            self.watercolor_img = self.original_img.copy()
            self.display_imgs()

    def load_secondary_img(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            secondary_img = cv2.imread(file_path)
            # Resize the new image to match the original image size
            self.secondary_img = cv2.resize(secondary_img, (self.original_img.shape[1], self.original_img.shape[0]))
            self.apply_watercolor_filter()

    def display_imgs(self):
        if self.original_img is not None:
            original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            modified_img_rgb = cv2.cvtColor(self.modified_img, cv2.COLOR_BGR2RGB)
            secondary_img_rgb = cv2.cvtColor(self.secondary_img, cv2.COLOR_BGR2RGB) if self.secondary_img is not None else None

            original_img_pil = Image.fromarray(original_img_rgb)
            modified_img_pil = Image.fromarray(modified_img_rgb)
            secondary_img_pil = Image.fromarray(secondary_img_rgb) if secondary_img_rgb is not None else None

            original_img_tk = ImageTk.PhotoImage(original_img_pil)
            modified_img_tk = ImageTk.PhotoImage(modified_img_pil)
            secondary_img_tk = ImageTk.PhotoImage(secondary_img_pil) if secondary_img_pil is not None else None

            self.original_panel.config(image=original_img_tk)
            self.original_panel.image = original_img_tk

            self.modified_panel.config(image=modified_img_tk)
            self.modified_panel.image = modified_img_tk

            if secondary_img_tk is not None:
                self.secondary_panel.config(image=secondary_img_tk)
                self.secondary_panel.image = secondary_img_tk

    def to_gray(self):
        self.hide_sliders()
        if self.original_img is not None:
            self.modified_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            self.modified_img = cv2.cvtColor(self.modified_img, cv2.COLOR_GRAY2BGR)
            self.display_imgs()

    def to_sepia(self):
        self.hide_sliders()
        if self.original_img is not None:
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            self.modified_img = cv2.transform(self.original_img, kernel)
            self.modified_img = np.clip(self.modified_img, 0, 255)
            self.display_imgs()

    def convert_hsv(self):
        if self.original_img is not None:
            hsv_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2HSV)
            hue_shift = self.hue_slider.get()
            h, s, v = cv2.split(hsv_img)
            h = (h + hue_shift) % 180  # Ensure hue values wrap around correctly
            hsv_img = cv2.merge([h, s, v])
            self.modified_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            self.display_imgs()

    def update_hsv(self, event):
        self.convert_hsv()

    def show_hsv_slider(self):
        self.hide_sliders()
        self.hue_slider.pack(side=LEFT, padx=5)
        self.hue_slider.bind("<Motion>", self.update_hsv)
        self.convert_hsv()

    def apply_median_blur(self):
        self.hide_sliders()
        if self.original_img is not None:
            self.modified_img = cv2.medianBlur(self.original_img, 5)
            self.display_imgs()

    def cartoon_filter(self):
        if self.original_img is not None:
            threshold = self.threshold_slider.get()
            gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.medianBlur(gray_img, 7)
            edge_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 9, threshold)
            color_img = cv2.bilateralFilter(self.original_img, 9, 300, 300)
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
            self.modified_img = cv2.bitwise_and(color_img, edge_img)
            self.display_imgs()

    def update_cartoon_filter(self, event):
        self.cartoon_filter()

    def show_cartoon_filter_slider(self):
        self.hide_sliders()
        self.threshold_slider.pack(side=LEFT, padx=5)
        self.threshold_slider.bind("<Motion>", self.update_cartoon_filter)
        self.cartoon_filter()

    def adjust_brightness_contrast(self):
        if self.original_img is not None:
            brightness = self.brightness_slider.get()
            contrast = self.contrast_slider.get()
            self.modified_img = cv2.convertScaleAbs(self.original_img, alpha=contrast, beta=brightness)
            self.display_imgs()

    def update_brightness_contrast(self, event):
        self.adjust_brightness_contrast()

    def show_brightness_contrast_sliders(self):
        self.hide_sliders()
        self.brightness_slider.pack(side=LEFT, padx=5)
        self.contrast_slider.pack(side=LEFT, padx=5)
        self.brightness_slider.bind("<Motion>", self.update_brightness_contrast)
        self.contrast_slider.bind("<Motion>", self.update_brightness_contrast)
        self.adjust_brightness_contrast()

    def show_filter_fields(self):
        self.hide_sliders()
        self.filter_matrix_entries = []
        for i in range(3):
            row = Frame(self.slider_frame)
            row.pack(side=TOP, padx=5, pady=2)
            row_entry = []
            for j in range(3):
                entry = Entry(row, width=5)
                entry.pack(side=LEFT)
                entry.insert(END, '0')
                row_entry.append(entry)
            self.filter_matrix_entries.append(row_entry)
        if self.apply_filter_button:
            self.apply_filter_button.pack_forget()
        self.apply_filter_button = Button(self.slider_frame, text="Apply Filter", command=self.apply_custom_filter)
        self.apply_filter_button.pack(side=TOP, pady=5)

    def apply_custom_filter(self):
        if self.original_img is not None:
            filter_matrix = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    filter_matrix[i, j] = float(self.filter_matrix_entries[i][j].get())
            self.modified_img = cv2.filter2D(self.original_img, -1, filter_matrix)
            self.display_imgs()

    def extract_channel(self, channel):
        self.hide_sliders()
        if self.original_img is not None:
            b, g, r = cv2.split(self.original_img)
            if channel == "Red":
                self.modified_img = r
            elif channel == "Green":
                self.modified_img = g
            elif channel == "Blue":
                self.modified_img = b
            self.modified_img = cv2.merge([self.modified_img] * 3)
            self.display_imgs()

    def apply_watercolor_filter(self):
        if self.original_img is not None and self.secondary_img is not None:
            mix_ratio = self.watercolor_slider.get() / 100.0
            self.watercolor_img = cv2.addWeighted(self.original_img, 1 - mix_ratio, self.secondary_img, mix_ratio, 0)
            self.modified_img = self.watercolor_img
            self.display_imgs()

    def update_watercolor_filter(self, event):
        self.apply_watercolor_filter()

    def show_watercolor_options(self):
        self.hide_sliders()
        self.watercolor_slider.pack(side=LEFT, padx=5)
        self.watercolor_slider.bind("<Motion>", self.update_watercolor_filter)

        if self.add_image_button:
            self.add_image_button.pack_forget()

        self.add_image_button = Button(self.slider_frame, text="Add Secondary Image", command=self.load_secondary_img)
        self.add_image_button.pack(side=TOP, pady=5)

        self.apply_watercolor_filter()

    def hide_sliders(self):
        for widget in self.slider_frame.winfo_children():
            widget.pack_forget()

        # Hide and remove filter matrix entries and apply button
        for row in self.filter_matrix_entries:
            for entry in row:
                entry.pack_forget()
        self.filter_matrix_entries = []

        if self.apply_filter_button:
            self.apply_filter_button.pack_forget()
            self.apply_filter_button = None

        if self.add_image_button:
            self.add_image_button.pack_forget()
            self.add_image_button = None


if __name__ == "__main__":
    master = Tk()
    app = ImageHandler(master)
    master.mainloop()