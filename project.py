import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
import cv2
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1200x800")

        # Left Panel for Algorithms (with Scrollbar)
        self.left_panel = tk.Frame(self.root, width=300, bg="lightgray")
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Add a Canvas and Scrollbar to the left panel
        self.canvas = tk.Canvas(self.left_panel, bg="lightgray")
        self.scrollbar = tk.Scrollbar(self.left_panel, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="lightgray")

        # Configure the canvas to work with the scrollbar
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right Panel for Images
        self.right_panel = tk.Frame(self.root, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Original Image Label
        self.original_image_label = tk.Label(self.right_panel, text="Original Image", bg="white")
        self.original_image_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Modified Image Label
        self.modified_image_label = tk.Label(self.right_panel, text="Modified Image", bg="white")
        self.modified_image_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Load Image Button
        self.load_button = tk.Button(self.scrollable_frame, text="Load Image", command=self.load_image, width=20, height=2)
        self.load_button.grid(row=0, column=0, columnspan=2, pady=10, padx=5, sticky="ew")

        # Reset Button
        self.reset_button = tk.Button(self.scrollable_frame, text="Reset Image", command=self.reset_image, width=20, height=2)
        self.reset_button.grid(row=1, column=0, columnspan=2, pady=5, padx=5, sticky="ew")

        # Save Image Button
        self.save_button = tk.Button(self.scrollable_frame, text="Save Image", command=self.save_image, width=20, height=2)
        self.save_button.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky="ew")

        # Group 1: Basic Image Operations
        basic_operations_frame = tk.LabelFrame(self.scrollable_frame, text="Basic Image Operations", bg="lightgray", font=("Arial", 10, "bold"))
        basic_operations_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        basic_operations = [
            ("Greyscale Conversion", self.greyscale_conversion),
            ("Negative Transformation", self.negative_transformation),
            ("Color Manipulation", self.color_manipulation),
            ("Flip Operations", self.flip_operations),
            ("Translation", self.translation),
            ("Scaling", self.scaling),
            ("Rotation", self.rotation),
            ("Cropping", self.cropping),
            ("Image Blending", self.image_blending),
            ("Brightness Adjustment", self.brightness_adjustment),
            ("Contrast Adjustment", self.contrast_adjustment),
            ("Color Filtering", self.color_filtering),
            ("Border and Padding", self.border_padding),
            ("Image Overlay", self.image_overlay),
        ]

        for idx, (algo_name, algo_func) in enumerate(basic_operations):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(basic_operations_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 2: Mathematical Operations on Images
        math_operations_frame = tk.LabelFrame(self.scrollable_frame, text="Mathematical Operations on Images", bg="lightgray", font=("Arial", 10, "bold"))
        math_operations_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        math_operations = [
            ("Pixel-wise Operations", self.mathematical_operations),
            ("Bitwise Operations", self.bitwise_operations_menu),
        ]

        for idx, (algo_name, algo_func) in enumerate(math_operations):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(math_operations_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 3: Transforms and Filtering in Image Processing
        transforms_frame = tk.LabelFrame(self.scrollable_frame, text="Transforms and Filtering in Image Processing", bg="lightgray", font=("Arial", 10, "bold"))
        transforms_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        transforms = [
            ("Fourier Transform", self.fourier_transform),
            ("Spatial Filters", self.spatial_filters),
            ("Edge Detection", self.edge_detection),
        ]

        for idx, (algo_name, algo_func) in enumerate(transforms):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(transforms_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 4: Image Enhancement
        enhancement_frame = tk.LabelFrame(self.scrollable_frame, text="Image Enhancement", bg="lightgray", font=("Arial", 10, "bold"))
        enhancement_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        enhancement = [
            ("Histogram Equalization", self.histogram_equalization),
            ("Contrast Stretching", self.contrast_stretching),
            ("Gamma Correction", self.gamma_correction),
        ]

        for idx, (algo_name, algo_func) in enumerate(enhancement):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(enhancement_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 5: Image Compression
        compression_frame = tk.LabelFrame(self.scrollable_frame, text="Image Compression", bg="lightgray", font=("Arial", 10, "bold"))
        compression_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        compression = [
            ("Lossless Compression", self.lossless_compression),
            ("Lossy Compression", self.lossy_compression),
        ]

        for idx, (algo_name, algo_func) in enumerate(compression):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(compression_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 6: Image Segmentation Using Thresholding and Clustering
        segmentation_frame = tk.LabelFrame(self.scrollable_frame, text="Image Segmentation Using Thresholding and Clustering", bg="lightgray", font=("Arial", 10, "bold"))
        segmentation_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        segmentation = [
            ("Thresholding", self.thresholding),
            ("K-means Clustering", self.kmeans_clustering),
        ]

        for idx, (algo_name, algo_func) in enumerate(segmentation):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(segmentation_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 7: Binary Image Processing
        binary_frame = tk.LabelFrame(self.scrollable_frame, text="Binary Image Processing", bg="lightgray", font=("Arial", 10, "bold"))
        binary_frame.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        binary = [
            ("Morphological Operations", self.morphological_operations),
            ("Boundary Extraction", self.boundary_extraction),
            ("Skeletonization", self.skeletonization),
        ]

        for idx, (algo_name, algo_func) in enumerate(binary):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(binary_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 8: Image Restoration
        restoration_frame = tk.LabelFrame(self.scrollable_frame, text="Image Restoration", bg="lightgray", font=("Arial", 10, "bold"))
        restoration_frame.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        restoration = [
            ("Noise Reduction", self.noise_reduction),
            ("Inpainting", self.inpainting),
        ]

        for idx, (algo_name, algo_func) in enumerate(restoration):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(restoration_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Group 9: Image Matching
        matching_frame = tk.LabelFrame(self.scrollable_frame, text="Image Matching", bg="lightgray", font=("Arial", 10, "bold"))
        matching_frame.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        matching = [
            ("Feature Detection", self.feature_detection),
            ("Template Matching", self.template_matching),
        ]

        for idx, (algo_name, algo_func) in enumerate(matching):
            row = idx // 2  # Split into 2 columns
            col = idx % 2
            button = tk.Button(matching_frame, text=algo_name, command=algo_func, width=20, height=2)
            button.grid(row=row, column=col, pady=2, padx=2, sticky="ew")

        # Configure the scrollable frame to expand buttons evenly
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        # Image Variables
        self.original_image = None
        self.modified_image = None
        self.tk_original_image = None
        self.tk_modified_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.modified_image = self.original_image.copy()
            self.display_images()

    def reset_image(self):
        if self.original_image:
            self.modified_image = self.original_image.copy()
            self.display_images()

    def save_image(self):
        if self.modified_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg")])
            if file_path:
                self.modified_image.save(file_path)
                messagebox.showinfo("Info", f"Image saved as {file_path}")

    def display_images(self):
        # Display Original Image
        self.tk_original_image = ImageTk.PhotoImage(self.original_image)
        self.original_image_label.config(image=self.tk_original_image)

        # Display Modified Image
        self.tk_modified_image = ImageTk.PhotoImage(self.modified_image)
        self.modified_image_label.config(image=self.tk_modified_image)

    def greyscale_conversion(self):
        if self.modified_image:
            self.modified_image = self.modified_image.convert("L")
            self.display_images()

    def negative_transformation(self):
        if self.modified_image:
            self.modified_image = ImageOps.invert(self.modified_image)
            self.display_images()

    def color_manipulation(self):
        if self.modified_image:
            r = simpledialog.askinteger("Red", "Enter Red intensity (0-255):", minvalue=0, maxvalue=255)
            g = simpledialog.askinteger("Green", "Enter Green intensity (0-255):", minvalue=0, maxvalue=255)
            b = simpledialog.askinteger("Blue", "Enter Blue intensity (0-255):", minvalue=0, maxvalue=255)
            if r is not None and g is not None and b is not None:
                img_array = np.array(self.modified_image)
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + r, 0, 255)  # Red channel
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] + g, 0, 255)  # Green channel
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] + b, 0, 255)  # Blue channel
                self.modified_image = Image.fromarray(img_array)
                self.display_images()

    def flip_operations(self):
        if self.modified_image:
            flip_window = tk.Toplevel(self.root)
            flip_window.title("Flip Operations")
            flip_window.geometry("200x100")

            horizontal_button = tk.Button(flip_window, text="Horizontal Flip", command=lambda: self.flip_image("horizontal"))
            horizontal_button.pack(pady=5)

            vertical_button = tk.Button(flip_window, text="Vertical Flip", command=lambda: self.flip_image("vertical"))
            vertical_button.pack(pady=5)

            diagonal_button = tk.Button(flip_window, text="Diagonal Flip", command=lambda: self.flip_image("diagonal"))
            diagonal_button.pack(pady=5)

    def flip_image(self, flip_type):
        if self.modified_image:
            if flip_type == "horizontal":
                self.modified_image = self.modified_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == "vertical":
                self.modified_image = self.modified_image.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_type == "diagonal":
                self.modified_image = self.modified_image.transpose(Image.FLIP_LEFT_RIGHT)
                self.modified_image = self.modified_image.transpose(Image.FLIP_TOP_BOTTOM)
            self.display_images()

    def translation(self):
        if self.modified_image:
            x_offset = simpledialog.askinteger("X Offset", "Enter horizontal offset:", minvalue=-1000, maxvalue=1000)
            y_offset = simpledialog.askinteger("Y Offset", "Enter vertical offset:", minvalue=-1000, maxvalue=1000)
            if x_offset is not None and y_offset is not None:
                self.modified_image = self.modified_image.transform(self.modified_image.size, Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
                self.display_images()

    def scaling(self):
        if self.modified_image:
            width = simpledialog.askinteger("Width", "Enter new width:", minvalue=1, maxvalue=5000)
            height = simpledialog.askinteger("Height", "Enter new height:", minvalue=1, maxvalue=5000)
            if width is not None and height is not None:
                self.modified_image = self.modified_image.resize((width, height))
                self.display_images()

    def rotation(self):
        if self.modified_image:
            angle = simpledialog.askinteger("Angle", "Enter rotation angle:", minvalue=-360, maxvalue=360)
            if angle is not None:
                rotation_window = tk.Toplevel(self.root)
                rotation_window.title("Rotation Direction")
                rotation_window.geometry("200x100")

                clockwise_button = tk.Button(rotation_window, text="Clockwise", command=lambda: self.rotate_image(angle))
                clockwise_button.pack(pady=5)

                counterclockwise_button = tk.Button(rotation_window, text="Counterclockwise", command=lambda: self.rotate_image(-angle))
                counterclockwise_button.pack(pady=5)

    def rotate_image(self, angle):
        if self.modified_image:
            self.modified_image = self.modified_image.rotate(angle)
            self.display_images()

    def cropping(self):
        if self.modified_image:
            left = simpledialog.askinteger("Left", "Enter left coordinate:", minvalue=0, maxvalue=self.modified_image.width)
            top = simpledialog.askinteger("Top", "Enter top coordinate:", minvalue=0, maxvalue=self.modified_image.height)
            right = simpledialog.askinteger("Right", "Enter right coordinate:", minvalue=0, maxvalue=self.modified_image.width)
            bottom = simpledialog.askinteger("Bottom", "Enter bottom coordinate:", minvalue=0, maxvalue=self.modified_image.height)
            if left is not None and top is not None and right is not None and bottom is not None:
                box = (left, top, right, bottom)
                self.modified_image = self.modified_image.crop(box)
                self.display_images()

    def image_blending(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                alpha = simpledialog.askfloat("Alpha", "Enter blending ratio (0.0 to 1.0):", minvalue=0.0, maxvalue=1.0)
                if alpha is not None:
                    blended_image = Image.blend(self.modified_image, second_image, alpha)
                    self.modified_image = blended_image
                    self.display_images()

    def brightness_adjustment(self):
        if self.modified_image:
            factor = simpledialog.askfloat("Brightness", "Enter brightness factor (0.0 to 2.0):", minvalue=0.0, maxvalue=2.0)
            if factor is not None:
                enhancer = ImageEnhance.Brightness(self.modified_image)
                self.modified_image = enhancer.enhance(factor)
                self.display_images()

    def contrast_adjustment(self):
        if self.modified_image:
            factor = simpledialog.askfloat("Contrast", "Enter contrast factor (0.0 to 2.0):", minvalue=0.0, maxvalue=2.0)
            if factor is not None:
                enhancer = ImageEnhance.Contrast(self.modified_image)
                self.modified_image = enhancer.enhance(factor)
                self.display_images()

    def color_filtering(self):
        if self.modified_image:
            # Example: Sepia Filter
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                    [0.349, 0.686, 0.168],
                                    [0.272, 0.534, 0.131]])
            img_array = np.array(self.modified_image)
            sepia_image = np.dot(img_array, sepia_filter.T)
            sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
            self.modified_image = Image.fromarray(sepia_image)
            self.display_images()

    def border_padding(self):
        if self.modified_image:
            border_padding_window = tk.Toplevel(self.root)
            border_padding_window.title("Border and Padding")
            border_padding_window.geometry("200x100")

            border_button = tk.Button(border_padding_window, text="Add Border", command=self.add_border)
            border_button.pack(pady=5)

            padding_button = tk.Button(border_padding_window, text="Add Padding", command=self.add_padding)
            padding_button.pack(pady=5)

    def add_border(self):
        if self.modified_image:
            border_size = simpledialog.askinteger("Border Size", "Enter border size:", minvalue=1, maxvalue=100)
            if border_size is not None:
                self.modified_image = ImageOps.expand(self.modified_image, border=border_size, fill="black")
                self.display_images()

    def add_padding(self):
        if self.modified_image:
            padding_size = simpledialog.askinteger("Padding Size", "Enter padding size:", minvalue=1, maxvalue=100)
            if padding_size is not None:
                self.modified_image = ImageOps.expand(self.modified_image, border=padding_size, fill="white")
                self.display_images()

    def image_overlay(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                overlay_image = Image.open(file_path)
                overlay_image = overlay_image.resize(self.modified_image.size)
                transparency = simpledialog.askfloat("Transparency", "Enter transparency (0.0 to 1.0):", minvalue=0.0, maxvalue=1.0)
                if transparency is not None:
                    x_offset = simpledialog.askinteger("X Offset", "Enter X position:", minvalue=0, maxvalue=self.modified_image.width)
                    y_offset = simpledialog.askinteger("Y Offset", "Enter Y position:", minvalue=0, maxvalue=self.modified_image.height)
                    if x_offset is not None and y_offset is not None:
                        overlay_image = overlay_image.convert("RGBA")
                        overlay_image.putalpha(int(255 * transparency))
                        self.modified_image.paste(overlay_image, (x_offset, y_offset), overlay_image)
                        self.display_images()

    def mathematical_operations(self):
        if self.modified_image:
            math_ops_window = tk.Toplevel(self.root)
            math_ops_window.title("Mathematical Operations")
            math_ops_window.geometry("200x100")

            pixelwise_button = tk.Button(math_ops_window, text="Pixel-wise Operations", command=self.pixelwise_operations_menu)
            pixelwise_button.pack(pady=5)

            bitwise_button = tk.Button(math_ops_window, text="Bitwise Operations", command=self.bitwise_operations_menu)
            bitwise_button.pack(pady=5)

    def pixelwise_operations_menu(self):
        if self.modified_image:
            pixelwise_window = tk.Toplevel(self.root)
            pixelwise_window.title("Pixel-wise Operations")
            pixelwise_window.geometry("200x200")

            addition_button = tk.Button(pixelwise_window, text="Addition", command=self.pixelwise_addition)
            addition_button.pack(pady=5)

            subtraction_button = tk.Button(pixelwise_window, text="Subtraction", command=self.pixelwise_subtraction)
            subtraction_button.pack(pady=5)

            multiplication_button = tk.Button(pixelwise_window, text="Multiplication", command=self.pixelwise_multiplication)
            multiplication_button.pack(pady=5)

            division_button = tk.Button(pixelwise_window, text="Division", command=self.pixelwise_division)
            division_button.pack(pady=5)

    def pixelwise_addition(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = np.clip(img_array1 + img_array2, 0, 255).astype(np.uint8)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def pixelwise_subtraction(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = np.clip(img_array1 - img_array2, 0, 255).astype(np.uint8)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def pixelwise_multiplication(self):
        if self.modified_image:
            img_array = np.array(self.modified_image)
            factor = simpledialog.askfloat("Multiplication Factor", "Enter multiplication factor:", minvalue=0.0, maxvalue=10.0)
            if factor is not None:
                result = np.clip(img_array * factor, 0, 255).astype(np.uint8)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def pixelwise_division(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = np.clip(img_array1 / (img_array2 + 1e-10), 0, 255).astype(np.uint8)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def bitwise_operations_menu(self):
        if self.modified_image:
            bitwise_window = tk.Toplevel(self.root)
            bitwise_window.title("Bitwise Operations")
            bitwise_window.geometry("200x200")

            and_button = tk.Button(bitwise_window, text="AND", command=self.bitwise_and)
            and_button.pack(pady=5)

            or_button = tk.Button(bitwise_window, text="OR", command=self.bitwise_or)
            or_button.pack(pady=5)

            xor_button = tk.Button(bitwise_window, text="XOR", command=self.bitwise_xor)
            xor_button.pack(pady=5)

            not_button = tk.Button(bitwise_window, text="NOT", command=self.bitwise_not)
            not_button.pack(pady=5)

    def bitwise_and(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = cv2.bitwise_and(img_array1, img_array2)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def bitwise_or(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = cv2.bitwise_or(img_array1, img_array2)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def bitwise_xor(self):
        if self.modified_image:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
            if file_path:
                second_image = Image.open(file_path)
                second_image = second_image.resize(self.modified_image.size)
                img_array1 = np.array(self.modified_image)
                img_array2 = np.array(second_image)
                result = cv2.bitwise_xor(img_array1, img_array2)
                self.modified_image = Image.fromarray(result)
                self.display_images()

    def bitwise_not(self):
        if self.modified_image:
            img_array = np.array(self.modified_image)
            result = cv2.bitwise_not(img_array)
            self.modified_image = Image.fromarray(result)
            self.display_images()

    def fourier_transform(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            self.modified_image = Image.fromarray(magnitude_spectrum.astype(np.uint8))
            self.display_images()

    def spatial_filters(self):
        if self.modified_image:
            self.modified_image = self.modified_image.filter(ImageFilter.BLUR)
            self.display_images()

    def edge_detection(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            edges = cv2.Canny(img_array, 100, 200)
            self.modified_image = Image.fromarray(edges)
            self.display_images()

    def histogram_equalization(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            equ = cv2.equalizeHist(img_array)
            self.modified_image = Image.fromarray(equ)
            self.display_images()

    def contrast_stretching(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            min_val = np.min(img_array)
            max_val = np.max(img_array)
            stretched = (img_array - min_val) * (255 / (max_val - min_val))
            self.modified_image = Image.fromarray(stretched.astype(np.uint8))
            self.display_images()

    def gamma_correction(self):
        if self.modified_image:
            gamma = simpledialog.askfloat("Gamma", "Enter gamma value:", minvalue=0.1, maxvalue=5.0)
            if gamma is not None:
                img_array = np.array(self.modified_image)
                corrected = np.power(img_array / 255.0, gamma) * 255
                self.modified_image = Image.fromarray(corrected.astype(np.uint8))
                self.display_images()

    def lossless_compression(self):
        if self.modified_image:
            self.modified_image.save("compressed_image.png", optimize=True)
            messagebox.showinfo("Info", "Image saved as compressed_image.png")

    def lossy_compression(self):
        if self.modified_image:
            self.modified_image.save("compressed_image.jpg", quality=50)
            messagebox.showinfo("Info", "Image saved as compressed_image.jpg")

    def thresholding(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            _, thresholded = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
            self.modified_image = Image.fromarray(thresholded)
            self.display_images()

    def kmeans_clustering(self):
        if self.modified_image:
            img_array = np.array(self.modified_image)
            Z = img_array.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img_array.shape))
            self.modified_image = Image.fromarray(res2)
            self.display_images()

    def morphological_operations(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            kernel = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(img_array, kernel, iterations=1)
            self.modified_image = Image.fromarray(erosion)
            self.display_images()

    def boundary_extraction(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary = np.zeros_like(img_array)
            cv2.drawContours(boundary, contours, -1, 255, 1)
            self.modified_image = Image.fromarray(boundary)
            self.display_images()

    def skeletonization(self):
        if self.modified_image:
            img_array = np.array(self.modified_image.convert("L"))
            size = np.size(img_array)
            skel = np.zeros(img_array.shape, np.uint8)
            ret, img = cv2.threshold(img_array, 127, 255, 0)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False
            while not done:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                zeros = size - cv2.countNonZero(img)
                if zeros == size:
                    done = True
            self.modified_image = Image.fromarray(skel)
            self.display_images()

    def noise_reduction(self):
        if self.modified_image:
            img_array = np.array(self.modified_image)
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            self.modified_image = Image.fromarray(denoised)
            self.display_images()

    def inpainting(self):
        if self.modified_image:
            img_array = np.array(self.modified_image)
            mask = np.zeros(img_array.shape[:2], np.uint8)
            mask[100:300, 100:300] = 255
            inpainted = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)
            self.modified_image = Image.fromarray(inpainted)
            self.display_images()

# Group 9: Image Matching Functions
def feature_detection(self):
    if self.modified_image:
        try:
            # Create detection method selection window
            method_window = tk.Toplevel(self.root)
            method_window.title("Select Detection Method")
            method_window.geometry("200x150")

            def apply_sift():
                self._apply_detection("SIFT")
                method_window.destroy()

            def apply_orb():
                self._apply_detection("ORB")
                method_window.destroy()

            # Add buttons for SIFT and ORB
            tk.Label(method_window, text="Choose Detection Method:").pack(pady=10)
            tk.Button(method_window, text="SIFT Detection", command=apply_sift).pack(pady=5)
            tk.Button(method_window, text="ORB Detection", command=apply_orb).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Error in feature detection: {str(e)}")

def feature_detection(self, method):
    try:
        # Convert PIL image to cv2 format
        img_array = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        if method == "SIFT":
            # Create SIFT detector
            detector = cv2.SIFT_create()
        else:
            # Create ORB detector
            detector = cv2.ORB_create(nfeatures=2000)

        # Detect and compute keypoints and descriptors
        keypoints, descriptors = detector.detectAndCompute(gray, None)

        # Draw keypoints
        if method == "SIFT":
            # For SIFT, use different color and size settings
            img_with_keypoints = cv2.drawKeypoints(
                img_array, 
                keypoints, 
                None,
                color=(0, 255, 0),  # Green color
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        else:
            # For ORB
            img_with_keypoints = cv2.drawKeypoints(
                img_array, 
                keypoints, 
                None,
                color=(255, 0, 0),  # Blue color
                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
            )

        # Convert back to PIL format
        self.modified_image = Image.fromarray(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        self.display_images()

        # Show results
        messagebox.showinfo("Detection Results", 
                          f"Found {len(keypoints)} features using {method}")

    except Exception as e:
        messagebox.showerror("Error", f"Detection failed: {str(e)}")

def template_matching(self):
    if self.modified_image:
        try:
            # Load template image
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
            )
            
            if file_path:
                # Convert main image to cv2 format
                main_img = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
                main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
                
                # Load and check template
                template = cv2.imread(file_path, 0)  # Read as grayscale
                if template is None:
                    raise ValueError("Failed to load template image")
                
                # Get template size
                h, w = template.shape
                
                # Create method selection window
                method_window = tk.Toplevel(self.root)
                method_window.title("Select Matching Method")
                
                def apply_method(method):
                    # Perform template matching
                    result = cv2.matchTemplate(main_gray, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    # Different handling for TM_SQDIFF methods
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                        
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    
                    # Draw rectangle
                    result_img = main_img.copy()
                    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
                    
                    # Convert and display
                    self.modified_image = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                    self.display_images()
                    
                    # Show match score
                    score = min_val if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_val
                    messagebox.showinfo("Match Result", f"Match score: {score:.4f}")
                    
                    method_window.destroy()
                
                # Add buttons for each method
                methods = [
                    ("TM_CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED),
                    ("TM_CCORR_NORMED", cv2.TM_CCORR_NORMED),
                    ("TM_SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED)
                ]
                
                tk.Label(method_window, text="Choose Matching Method:").pack(pady=5)
                
                for name, method in methods:
                    tk.Button(
                        method_window, 
                        text=name,
                        command=lambda m=method: apply_method(m)
                    ).pack(pady=2)
                
        except Exception as e:
            messagebox.showerror("Error", f"Template matching failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
