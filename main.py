

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import cv2
import numpy as np
from scipy import ndimage, fftpack
from skimage import feature, morphology, restoration, segmentation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing Application")
        self.setup_ui()
        self.current_image = None
        self.original_image = None
        self.history = []

    def setup_ui(self):
        # Main container
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel for tools
        self.tools_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.tools_frame)

        # Right panel for image display
        self.display_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.display_frame)

        self.setup_tools()
        self.setup_display()

    def setup_tools(self):
        # File operations
        file_frame = ttk.LabelFrame(self.tools_frame, text="File Operations")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="Save Image", command=self.save_image).pack(fill=tk.X, padx=5, pady=2)

        # Basic operations
        basic_frame = ttk.LabelFrame(self.tools_frame, text="Basic Operations")
        basic_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(basic_frame, text="Greyscale", command=self.convert_greyscale).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(basic_frame, text="Negative", command=self.negative_transform).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(basic_frame, text="Flip Horizontal", command=self.flip_horizontal).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(basic_frame, text="Flip Vertical", command=self.flip_vertical).pack(fill=tk.X, padx=5, pady=2)

        # Enhancement operations
        enhance_frame = ttk.LabelFrame(self.tools_frame, text="Enhancement")
        enhance_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(enhance_frame, text="Brightness:").pack(padx=5)
        self.brightness_scale = ttk.Scale(enhance_frame, from_=0, to=2, orient=tk.HORIZONTAL)
        self.brightness_scale.set(1)
        self.brightness_scale.pack(fill=tk.X, padx=5)
        self.brightness_scale.bind("<ButtonRelease-1>", self.adjust_brightness)

    def setup_display(self):
        self.canvas = tk.Canvas(self.display_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.current_image = self.original_image.copy()
            self.display_image()
            self.history = [self.current_image.copy()]

    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                self.current_image.save(file_path)

    def display_image(self):
        if self.current_image:
            # Resize image to fit canvas while maintaining aspect ratio
            display_size = (800, 600)
            self.current_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for display
            self.photo = ImageTk.PhotoImage(self.current_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update canvas scrollregion
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    # Basic Image Operations
    def convert_greyscale(self):
        if self.current_image:
            self.current_image = self.current_image.convert('L')
            self.display_image()
            self.history.append(self.current_image.copy())

    def negative_transform(self):
        if self.current_image:
            self.current_image = Image.eval(self.current_image, lambda x: 255 - x)
            self.display_image()
            self.history.append(self.current_image.copy())

    def flip_horizontal(self):
        if self.current_image:
            self.current_image = self.current_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            self.display_image()
            self.history.append(self.current_image.copy())

    def flip_vertical(self):
        if self.current_image:
            self.current_image = self.current_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            self.display_image()
            self.history.append(self.current_image.copy())

    def adjust_brightness(self, event=None):
        if self.current_image:
            factor = self.brightness_scale.get()
            enhancer = ImageEnhance.Brightness(self.current_image)
            self.current_image = enhancer.enhance(factor)
            self.display_image()
            self.history.append(self.current_image.copy())

    # Additional methods for other operations...
    def apply_gaussian_blur(self):
        if self.current_image:
            self.current_image = self.current_image.filter(ImageFilter.GaussianBlur(radius=2))
            self.display_image()
            self.history.append(self.current_image.copy())

    def apply_edge_detection(self):
        if self.current_image:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(self.current_image)
            edges = cv2.Canny(img_array, 100, 200)
            self.current_image = Image.fromarray(edges)
            self.display_image()
            self.history.append(self.current_image.copy())

    def apply_histogram_equalization(self):
        if self.current_image:
            if self.current_image.mode != 'L':
                self.current_image = self.current_image.convert('L')
            img_array = np.array(self.current_image)
            img_eq = cv2.equalizeHist(img_array)
            self.current_image = Image.fromarray(img_eq)
            self.display_image()
            self.history.append(self.current_image.copy())

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()