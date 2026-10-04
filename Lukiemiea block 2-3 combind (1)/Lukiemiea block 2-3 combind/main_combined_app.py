import os
import sys
import threading
import queue
import traceback
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Frame, Label, Button, IntVar, DoubleVar

# --- HDS Denoising Imports and Parameters ---
LAMBDA = 0.05
DT = 0.15
G_PARAM = 1.3
H_PARAM = 1.0
EPSILON = 1e-8

# --- HDS Denoising Functions (from hds.py) ---
def compute_gradients_and_coefficients(image, H_param, G_param, epsilon):
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    grad_u_W = image - padded_image[1:-1, :-2]
    grad_u_N = image - padded_image[:-2, 1:-1]
    grad_u_S = padded_image[2:, 1:-1] - image
    grad_u_E = padded_image[1:-1, 2:] - image
    c_N = G_param * (1.0 / (1.0 + (grad_u_N**2) / (H_param**2)) + 1.0 / (np.abs(grad_u_N) + epsilon))
    c_S = G_param * (1.0 / (1.0 + (grad_u_S**2) / (H_param**2)) + 1.0 / (np.abs(grad_u_S) + epsilon))
    c_W = G_param * (1.0 / (1.0 + (grad_u_W**2) / (H_param**2)) + 1.0 / (np.abs(grad_u_W) + epsilon))
    c_E = G_param * (1.0 / (1.0 + (grad_u_E**2) / (H_param**2)) + 1.0 / (np.abs(grad_u_E) + epsilon))
    return grad_u_W, grad_u_N, grad_u_S, grad_u_E, c_N, c_S, c_W, c_E

def hybrid_denoise(noisy_image_float, original_f_float, iterations=200, progress_callback=None):
    u = noisy_image_float.copy()
    f = original_f_float.copy()
    for n in range(iterations):
        if progress_callback and n % 10 == 0:
            progress_callback(n, iterations)
        grad_u_W, grad_u_N, grad_u_S, grad_u_E, c_N, c_S, c_W, c_E = \
            compute_gradients_and_coefficients(u, H_PARAM, G_PARAM, EPSILON)
        div_term = np.zeros_like(u)
        div_term += c_N * grad_u_N
        div_term += c_S * grad_u_S
        div_term += c_W * grad_u_W
        div_term += c_E * grad_u_E
        log_prior_term = LAMBDA * ((u / (f + EPSILON)) - 1.0) * (1.0 / (u + EPSILON))
        u_next = u + DT * (div_term - log_prior_term)
        if np.isnan(u_next).any() or np.isinf(u_next).any():
            u = np.clip(u, 0.0, 1.0)
            break
        u = np.clip(u_next, 0.0, 1.0)
    if progress_callback:
        progress_callback(iterations, iterations)
    return u

def process_color_image(img_bgr, iterations=300, progress_callback=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r_channel = img_rgb[:, :, 0].astype(np.float64) / 255.0
    g_channel = img_rgb[:, :, 1].astype(np.float64) / 255.0
    b_channel = img_rgb[:, :, 2].astype(np.float64) / 255.0
    def channel_progress(current, total, channel_name, channel_num):
        if progress_callback:
            overall_progress = (channel_num - 1) * iterations + current
            overall_total = 3 * iterations
            progress_callback(overall_progress, overall_total, f"Processing {channel_name} channel...")
    if progress_callback:
        progress_callback(0, 3 * iterations, "Processing Red channel...")
    r_denoised = hybrid_denoise(r_channel, r_channel.copy(), iterations,
                                lambda c, t: channel_progress(c, t, "Red", 1))
    if progress_callback:
        progress_callback(iterations, 3 * iterations, "Processing Green channel...")
    g_denoised = hybrid_denoise(g_channel, g_channel.copy(), iterations,
                                lambda c, t: channel_progress(c, t, "Green", 2))
    if progress_callback:
        progress_callback(2 * iterations, 3 * iterations, "Processing Blue channel...")
    b_denoised = hybrid_denoise(b_channel, b_channel.copy(), iterations,
                                lambda c, t: channel_progress(c, t, "Blue", 3))
    denoised_rgb = np.stack([r_denoised, g_denoised, b_denoised], axis=2)
    denoised_rgb = (denoised_rgb * 255).astype(np.uint8)
    if progress_callback:
        progress_callback(3 * iterations, 3 * iterations, "Combining channels...")
    return denoised_rgb

# --- Edge Detection Imports and Classes (from app.py) ---

from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sobel, prewitt, roberts, scharr, laplace
from skimage.morphology import remove_small_objects, binary_closing, disk

# --- EdgeDetection and SimpleEdgeDetectionGUI from app.py ---
class EdgeDetection:
    """
    A class for detecting edges in images using various methods.
    """
    def __init__(self):
        pass
    def preprocess_image(self, image):
        # Convert to grayscale float if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float64)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
        return image
    def adaptive_principal_curvature(self, image, sigma=1.0, alpha=0.5):
        hessian = hessian_matrix(image, sigma=sigma, order='rc')
        eigvals = hessian_matrix_eigvals(hessian)
        response = np.abs(eigvals[0]) + alpha * np.abs(eigvals[1])
        response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        return response
    def central_difference(self, image):
        dx = np.zeros_like(image)
        dy = np.zeros_like(image)
        dx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
        dy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
        response = np.hypot(dx, dy)
        response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        return response
    def canny_edge_detection(self, image, low_threshold=0.1, high_threshold=0.2):
        from skimage.feature import canny
        edges = canny(image, low_threshold=low_threshold, high_threshold=high_threshold)
        return edges.astype(np.float64)
    def laplacian_of_gaussian(self, image, sigma=1.0):
        smoothed = gaussian_filter(image, sigma=sigma)
        log_response = laplace(smoothed)
        log_response = np.abs(log_response)
        log_response = (log_response - np.min(log_response)) / (np.max(log_response) - np.min(log_response) + 1e-10)
        return log_response
    def detect_edges(self, image, method="Sobel", sigma=1.0, alpha=0.5, threshold=0.3,
                    post_process=True, min_size=20, closing_radius=2,
                    canny_low=0.1, canny_high=0.2):
        preprocessed = self.preprocess_image(image)
        if method.upper() != "CANNY":
            img = gaussian_filter(preprocessed, sigma=sigma)
        else:
            img = preprocessed
        if method.upper() == "APC":
            response = self.adaptive_principal_curvature(img, sigma=sigma, alpha=alpha)
        elif method.upper() == "SOBEL":
            response = sobel(img)
        elif method.upper() == "PREWITT":
            response = prewitt(img)
        elif method.upper() == "ROBERTS":
            response = roberts(img)
        elif method.upper() == "SCHARR":
            response = scharr(img)
        elif method.upper() == "CENTRALDIFF":
            response = self.central_difference(img)
        elif method.upper() == "CANNY":
            response = self.canny_edge_detection(img, low_threshold=canny_low, high_threshold=canny_high)
        elif method.upper() == "LOG":
            response = self.laplacian_of_gaussian(img, sigma=sigma)
        else:
            response = sobel(img)
        binary = response > threshold
        if post_process and method.upper() != "CANNY":
            binary = remove_small_objects(binary, min_size=min_size)
            binary = binary_closing(binary, disk(closing_radius))
        return binary, response
    def create_focused_edge_views(self, image, binary_mask, response):
        if len(image.shape) == 2:
            rgb_image = np.stack([image] * 3, axis=2)
        else:
            rgb_image = image.copy()
        edge_overlay = rgb_image.copy()
        if len(edge_overlay.shape) == 3:
            edge_overlay[binary_mask, 0] = np.minimum(255, edge_overlay[binary_mask, 0] * 0.5)
            edge_overlay[binary_mask, 1] = np.minimum(255, edge_overlay[binary_mask, 1] * 0.5 + 128)
            edge_overlay[binary_mask, 2] = np.minimum(255, edge_overlay[binary_mask, 2] * 0.5 + 128)
        edge_map = np.zeros_like(rgb_image)
        edge_map[binary_mask] = [255, 255, 255]
        response_normalized = (response * 255).astype(np.uint8)
        response_colored = cv2.applyColorMap(response_normalized, cv2.COLORMAP_JET)
        response_colored = cv2.cvtColor(response_colored, cv2.COLOR_BGR2RGB)
        enhanced = np.zeros_like(rgb_image)
        enhanced[binary_mask] = rgb_image[binary_mask]
        if len(enhanced.shape) == 3 and np.any(binary_mask):
            for c in range(3):
                enhanced[..., c][binary_mask] = np.clip(enhanced[..., c][binary_mask] * 1.5, 0, 255)
        return edge_overlay, edge_map, response_colored, enhanced

def load_image(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def save_image(file_path, image):
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(file_path, image)
        return success
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

class SimpleEdgeDetectionGUI:
    def __init__(self, root):
        self.root = root
        # Only set title/geometry if root is a Tk window (not a Frame)
        if isinstance(self.root, tk.Tk):
            self.root.title("Edge Detection Tool")
            self.root.geometry("1200x700")
        self.detector = EdgeDetection()
        self.image = None
        self.file_path = None
        self.binary_mask = None
        self.response = None
        self.edge_overlay = None
        self.edge_map = None
        self.response_colored = None
        self.enhanced = None
        self.current_method = None
        self.create_widgets()
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.LabelFrame(main_frame, text="Edge Detection Controls")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        file_buttons_frame = ttk.Frame(control_frame)
        file_buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(file_buttons_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        method_frame = ttk.LabelFrame(control_frame, text="Edge Detection Methods")
        method_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        methods_info = [
            ("Sobel", "Sobel edge detection"),
            ("Prewitt", "Prewitt edge detection"),
            ("Roberts", "Roberts cross-gradient"),
            ("Scharr", "Scharr edge detection"),
            ("CentralDiff", "Central Difference"),
            ("Canny", "Canny edge detection"),
            ("LOG", "Laplacian of Gaussian"),
            ("APC", "Adaptive Principal Curvature")
        ]
        for i, (method, tooltip) in enumerate(methods_info):
            row = i // 4
            col = i % 4
            btn = ttk.Button(method_frame, text=method,
                           command=lambda m=method: self.apply_method(m),
                           width=12)
            btn.grid(row=row, column=col, padx=3, pady=3, sticky=tk.W+tk.E)
        for i in range(4):
            method_frame.columnconfigure(i, weight=1)
        params_frame = ttk.LabelFrame(control_frame, text="Parameters")
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(params_frame, text="Current Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.current_method_var = tk.StringVar(value="None")
        ttk.Label(params_frame, textvariable=self.current_method_var,
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(params_frame, text="Threshold:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_entry = ttk.Entry(params_frame, textvariable=self.threshold_var, width=5)
        threshold_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        ttk.Label(params_frame, text="Sigma:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.sigma_var = tk.DoubleVar(value=1.0)
        sigma_entry = ttk.Entry(params_frame, textvariable=self.sigma_var, width=5)
        sigma_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        ttk.Label(params_frame, text="View:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.view_var = tk.StringVar(value="original")
        view_combo = ttk.Combobox(params_frame, textvariable=self.view_var,
                                values=["original", "edge_overlay", "edge_map", "response", "enhanced"],
                                width=12)
        view_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(params_frame, text="Update", command=self.update_current_method).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.status_var = tk.StringVar(value="Ready - Load an image to start")
        self.current_status = "Ready - Load an image to start"
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.image_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.image_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_frame = ttk.LabelFrame(self.image_paned, text="Original Image")
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        self.image_paned.add(self.original_frame, width=400)
        self.processed_frame = ttk.LabelFrame(self.image_paned, text="Processed Image")
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True)
        self.image_paned.add(self.processed_frame, width=400)
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.file_path = file_path
            self.image = load_image(file_path)
            if self.image is not None:
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.current_status = f"Loaded: {os.path.basename(file_path)}"
                self.display_image(self.image, panel="original")
                self.binary_mask = None
                self.response = None
                self.edge_overlay = None
                self.edge_map = None
                self.response_colored = None
                self.enhanced = None
                self.current_method = None
                self.current_method_var.set("None")
            else:
                self.status_var.set("Failed to load image")
                self.current_status = "Failed to load image"
    def apply_method(self, method):
        if self.image is None:
            self.status_var.set("Please load an image first")
            self.current_status = "Please load an image first"
            return
        self.current_method = method
        self.current_method_var.set(method)
        threshold = float(self.threshold_var.get())
        sigma = float(self.sigma_var.get())
        alpha = 0.5
        post_process = True
        min_size = 20
        closing_radius = 2
        canny_low = 0.1
        canny_high = 0.2
        self.binary_mask, self.response = self.detector.detect_edges(
            self.image, method=method, sigma=sigma, alpha=alpha,
            threshold=threshold, post_process=post_process,
            min_size=min_size, closing_radius=closing_radius,
            canny_low=canny_low, canny_high=canny_high
        )
        self.edge_overlay, self.edge_map, self.response_colored, self.enhanced = self.detector.create_focused_edge_views(
            self.image, self.binary_mask, self.response
        )
        self.view_var.set("response")
        self.update_view()
        status_msg = f"{method} applied - Threshold: {threshold:.2f}, Sigma: {sigma:.1f}"
        self.status_var.set(status_msg)
        self.current_status = status_msg
    def update_current_method(self):
        if self.current_method and self.image is not None:
            self.apply_method(self.current_method)
        else:
            self.status_var.set("No method selected or no image loaded")
            self.current_status = "No method selected or no image loaded"
    def process_image(self):
        if self.current_method:
            self.apply_method(self.current_method)
        else:
            self.status_var.set("Please select a method by clicking one of the method buttons")
            self.current_status = "Please select a method by clicking one of the method buttons"
    def update_view(self, event=None):
        if self.image is None:
            return
        view = self.view_var.get()
        self.display_image(self.image, panel="original")
        if view == "original":
            self.display_image(self.image, panel="processed")
        elif view == "edge_overlay" and self.edge_overlay is not None:
            self.display_image(self.edge_overlay, panel="processed")
        elif view == "edge_map" and self.edge_map is not None:
            self.display_image(self.edge_map, panel="processed")
        elif view == "response" and self.response_colored is not None:
            self.display_image(self.response_colored, panel="processed")
        elif view == "enhanced" and self.enhanced is not None:
            self.display_image(self.enhanced, panel="processed")
        else:
            self.display_image(self.image, panel="processed")
    def display_image(self, img, panel="original"):
        h, w = img.shape[:2]
        max_h = 550
        max_w = 400
        if h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
        else:
            img_resized = img.copy()
        img_display = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        img_display = cv2.imencode('.png', img_display)[1].tobytes()
        img_tk = tk.PhotoImage(data=img_display)
        if panel == "original":
            self.original_label.configure(image=img_tk)
            self.original_label.image = img_tk
        else:
            self.processed_label.configure(image=img_tk)
            self.processed_label.image = img_tk
    def save_results(self):
        if self.image is None or self.binary_mask is None:
            self.status_var.set("Process an image before saving")
            self.current_status = "Process an image before saving"
            return
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        method = self.current_method or "unknown"
        original_path = os.path.join(save_dir, f"{base_filename}_original.png")
        save_image(original_path, self.image)
        if self.edge_overlay is not None:
            overlay_path = os.path.join(save_dir, f"{base_filename}_{method}_overlay.png")
            save_image(overlay_path, self.edge_overlay)
        if self.edge_map is not None:
            map_path = os.path.join(save_dir, f"{base_filename}_{method}_edges.png")
            save_image(map_path, self.edge_map)
        if self.response_colored is not None:
            response_path = os.path.join(save_dir, f"{base_filename}_{method}_response.png")
            save_image(response_path, self.response_colored)
        if self.enhanced is not None:
            enhanced_path = os.path.join(save_dir, f"{base_filename}_{method}_enhanced.png")
            save_image(enhanced_path, self.enhanced)
        status_msg = f"Results saved to {save_dir}"
        self.status_var.set(status_msg)
        self.current_status = status_msg

# --- AdvancedDenoisingApp from hds.py ---
class AdvancedDenoisingApp:


    def __init__(self, master):
        self.master = master
        if isinstance(self.master, tk.Tk):
            self.master.title("Block 2 : HDS edge detection")
            self.master.geometry("1400x900")
            self.master.configure(bg='#f0f0f0')
        self.iterations_var = IntVar(value=300)
        self.lambda_var = DoubleVar(value=LAMBDA)
        self.dt_var = DoubleVar(value=DT)
        self.g_param_var = DoubleVar(value=G_PARAM)
        self.h_param_var = DoubleVar(value=H_PARAM)
        self.current_file_path = None
        self.original_image = None
        self.denoised_image = None
        self.is_color = False
        self.processing_thread = None
        self.result_queue = queue.Queue()
        self.setup_ui()
        self.check_queue()


    def setup_ui(self):
        # Main container
        main_frame = Frame(self.master, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        # Title
        title_label = Label(
            main_frame,
            text="HDS Edge Detection & Denoising",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#1a237e',
            pady=10
        )
        title_label.pack(pady=(0, 5))
        subtitle_label = Label(
            main_frame,
            text="A Hybrid Diffusion-Steered (HDS) method combining Total Variation and Perona-Malik diffusivities\nfor robust edge detection and denoising beyond classical techniques.",
            font=('Arial', 11, 'italic'),
            bg='#f0f0f0',
            fg='#333366',
            justify='center',
            wraplength=900
        )
        subtitle_label.pack(pady=(0, 18))
        # Control Panel
        control_frame = Frame(main_frame, bg='#ffffff', relief='raised', bd=2)
        control_frame.pack(fill='x', pady=(0, 10))
        # File operations
        file_frame = Frame(control_frame, bg='#ffffff')
        file_frame.pack(fill='x', padx=10, pady=10)
        Label(file_frame, text="File Operations:", font=('Arial', 10, 'bold'),
              bg='#ffffff').pack(anchor='w')
        button_frame = Frame(file_frame, bg='#ffffff')
        button_frame.pack(fill='x', pady=5)
        self.upload_button = Button(button_frame, text="📁 Upload Image",
                                   command=self.upload_image, bg='#4CAF50', fg='white',
                                   font=('Arial', 10, 'bold'), padx=20)
        self.upload_button.pack(side='left', padx=(0, 10))
        self.save_button = Button(button_frame, text="💾 Save Results",
                                 command=self.save_results, bg='#2196F3', fg='white',
                                 font=('Arial', 10, 'bold'), padx=20, state='disabled')
        self.save_button.pack(side='left', padx=(0, 10))
        self.process_button = Button(button_frame, text="🔄 Process Image",
                                    command=self.process_image, bg='#FF9800', fg='white',
                                    font=('Arial', 10, 'bold'), padx=20, state='disabled')
        self.process_button.pack(side='left')
        # Parameter controls
        param_frame = Frame(control_frame, bg='#ffffff')
        param_frame.pack(fill='x', padx=10, pady=(0, 10))
        Label(param_frame, text="Algorithm Parameters:", font=('Arial', 10, 'bold'),
              bg='#ffffff').pack(anchor='w')
        param_grid = Frame(param_frame, bg='#ffffff')
        param_grid.pack(fill='x', pady=5)
        Label(param_grid, text="Iterations:", bg='#ffffff').grid(row=0, column=0, sticky='w', padx=(0, 5))
        iterations_scale = ttk.Scale(param_grid, from_=50, to=500, variable=self.iterations_var,
                                    orient='horizontal', length=100)
        iterations_scale.grid(row=0, column=1, sticky='ew', padx=5)
        self.iterations_label = Label(param_grid, text=str(self.iterations_var.get()), bg='#ffffff')
        self.iterations_label.grid(row=0, column=2, padx=5)
        Label(param_grid, text="Lambda:", bg='#ffffff').grid(row=0, column=3, sticky='w', padx=(20, 5))
        lambda_scale = ttk.Scale(param_grid, from_=0.01, to=0.2, variable=self.lambda_var,
                                orient='horizontal', length=100)
        lambda_scale.grid(row=0, column=4, sticky='ew', padx=5)
        self.lambda_label = Label(param_grid, text=f"{self.lambda_var.get():.3f}", bg='#ffffff')
        self.lambda_label.grid(row=0, column=5, padx=5)
        iterations_scale.configure(command=lambda v: self.iterations_label.config(text=str(int(float(v)))))
        lambda_scale.configure(command=lambda v: self.lambda_label.config(text=f"{float(v):.3f}"))
        status_frame = Frame(control_frame, bg='#ffffff')
        status_frame.pack(fill='x', padx=10, pady=(0, 10))
        self.status_label = Label(status_frame, text="Ready to process image",
                                 font=('Arial', 10), bg='#ffffff', fg='#666666')
        self.status_label.pack(anchor='w')
        progress_frame = Frame(status_frame, bg='#ffffff')
        progress_frame.pack(fill='x', pady=5)
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress.pack(side='left', fill='x', expand=True)
        self.progress_label = Label(progress_frame, text="0%", bg='#ffffff', width=5)
        self.progress_label.pack(side='right', padx=(5, 0))
        self.cancel_button = Button(progress_frame, text="Cancel",
                                   command=self.cancel_processing, bg='#f44336', fg='white',
                                   font=('Arial', 9), state='disabled')
        self.cancel_button.pack(side='right', padx=(5, 5))
        display_frame = Frame(main_frame, bg='#ffffff', relief='raised', bd=2)
        display_frame.pack(fill='both', expand=True)
        images_frame = Frame(display_frame, bg='#ffffff')
        images_frame.pack(fill='both', expand=True, padx=10, pady=10)
        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)
        images_frame.grid_columnconfigure(2, weight=1)
        images_frame.grid_rowconfigure(0, weight=1)
        orig_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        orig_panel.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        Label(orig_panel, text="Original Image", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.original_display = Label(orig_panel, bg='#f8f8f8')
        self.original_display.pack(expand=True)
        denoised_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        denoised_panel.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        Label(denoised_panel, text="Denoised Image", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.denoised_display = Label(denoised_panel, bg='#f8f8f8')
        self.denoised_display.pack(expand=True)
        diff_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        diff_panel.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        Label(diff_panel, text="Noise Removed", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.diff_display = Label(diff_panel, bg='#f8f8f8')
        self.diff_display.pack(expand=True)
        info_frame = Frame(display_frame, bg='#ffffff')
        info_frame.pack(fill='x', padx=10, pady=(0, 10))
        self.info_label = Label(
            info_frame,
            text="Load an image to begin analysis with the advanced HDS edge detection and denoising method.",
            font=('Arial', 10),
            bg='#ffffff',
            fg='#666666',
            wraplength=900,
            justify='left'
        )
        self.info_label.pack(anchor='w')

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.current_file_path = file_path
            self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image from {file_path}")
            self.is_color = (len(img.shape) == 3 and img.shape[2] == 3)
            self.original_image = img
            self.denoised_image = None
            self.clear_displays()
            self.display_image_in_label(img, self.original_display)
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.info_label.config(text="Adjust parameters and click 'Process Image' to denoise.")
            self.process_button.config(state='normal')
            self.save_button.config(state='disabled')
        except Exception as e:
            self.status_label.config(text=f"Error loading image: {str(e)}")
            self.info_label.config(text="Failed to load image.")

    def process_image(self):
        if self.original_image is None:
            self.status_label.config(text="Please upload an image first.")
            return
        if self.processing_thread and self.processing_thread.is_alive():
            self.status_label.config(text="Processing already in progress.")
            return
        self.progress['value'] = 0
        self.progress_label.config(text="0%")
        self.status_label.config(text="Processing image...")
        self.info_label.config(text="Denoising in progress...")
        self.process_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        self.save_button.config(state='disabled')
        self.result_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_image_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_image_thread(self):
        try:
            iterations = int(self.iterations_var.get())
            lambda_ = float(self.lambda_var.get())
            dt = float(self.dt_var.get())
            g_param = float(self.g_param_var.get())
            h_param = float(self.h_param_var.get())
            img = self.original_image
            if self.is_color:
                def progress_callback(current, total, msg=None):
                    percent = int(100 * current / max(1, total))
                    self.result_queue.put(("progress", percent, msg))
                denoised = process_color_image(img, iterations=iterations, progress_callback=progress_callback)
            else:
                img_float = img.astype(np.float64) / 255.0
                def progress_callback(current, total, msg=None):
                    percent = int(100 * current / max(1, total))
                    self.result_queue.put(("progress", percent, msg))
                denoised_float = hybrid_denoise(img_float, img_float.copy(), iterations=iterations, progress_callback=progress_callback)
                denoised = (denoised_float * 255).astype(np.uint8)
            self.result_queue.put(("done", denoised))
        except Exception as e:
            tb = traceback.format_exc()
            self.result_queue.put(("error", str(e), tb))

    def cancel_processing(self):
        # Not a true cancel, but disables UI and resets
        self.reset_processing_ui()
        self.status_label.config(text="Processing cancelled.")
        self.info_label.config(text="Processing was cancelled.")

    def reset_processing_ui(self):
        self.progress['value'] = 0
        self.progress_label.config(text="0%")
        self.process_button.config(state='normal')
        self.cancel_button.config(state='disabled')
        self.save_button.config(state='disabled')

    def check_queue(self):
        try:
            while not self.result_queue.empty():
                msg = self.result_queue.get_nowait()
                if msg[0] == "progress":
                    percent = msg[1]
                    self.progress['value'] = percent
                    self.progress_label.config(text=f"{percent}%")
                    if msg[2]:
                        self.status_label.config(text=msg[2])
                elif msg[0] == "done":
                    self.denoised_image = msg[1]
                    self.display_results()
                    self.reset_processing_ui()
                    self.status_label.config(text="Processing complete.")
                    self.info_label.config(text="Denoising complete. You can save the results.")
                    self.save_button.config(state='normal')
                elif msg[0] == "error":
                    self.status_label.config(text=f"Error: {msg[1]}")
                    self.info_label.config(text="An error occurred during processing.")
                    print(msg[2])
                    self.reset_processing_ui()
        except Exception as e:
            print(f"Error in check_queue: {e}")
        finally:
            self.master.after(100, self.check_queue)

    def display_results(self):
        if self.original_image is not None:
            self.display_image_in_label(self.original_image, self.original_display)
        if self.denoised_image is not None:
            self.display_image_in_label(self.denoised_image, self.denoised_display)
            # Show difference image
            try:
                if self.is_color:
                    diff = cv2.absdiff(self.original_image, self.denoised_image)
                else:
                    diff = cv2.absdiff(self.original_image, self.denoised_image)
                self.display_image_in_label(diff, self.diff_display)
            except Exception as e:
                self.info_label.config(text=f"Error displaying diff: {e}")

    def display_image_in_label(self, image, label):
        try:
            if len(image.shape) == 2:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            max_h, max_w = 350, 350
            scale = min(max_h / h, max_w / w, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            label.configure(image=img_tk)
            label.image = img_tk
        except Exception as e:
            label.configure(text=f"Error displaying image: {e}")

    def save_results(self):
        if self.denoised_image is None:
            self.status_label.config(text="No denoised image to save.")
            return
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
        base_filename = os.path.splitext(os.path.basename(self.current_file_path))[0]
        orig_path = os.path.join(save_dir, f"{base_filename}_original.png")
        denoised_path = os.path.join(save_dir, f"{base_filename}_denoised.png")
        diff_path = os.path.join(save_dir, f"{base_filename}_diff.png")
        try:
            cv2.imwrite(orig_path, self.original_image)
            cv2.imwrite(denoised_path, self.denoised_image)
            if self.is_color:
                diff = cv2.absdiff(self.original_image, self.denoised_image)
            else:
                diff = cv2.absdiff(self.original_image, self.denoised_image)
            cv2.imwrite(diff_path, diff)
            self.status_label.config(text=f"Results saved to {save_dir}")
            self.info_label.config(text="Results saved successfully.")
        except Exception as e:
            self.status_label.config(text=f"Error saving: {e}")
            self.info_label.config(text="Failed to save results.")

    def clear_displays(self):
        for lbl in [self.original_display, self.denoised_display, self.diff_display]:
            lbl.config(image='', text='')
            lbl.image = None

# --- Combined Main Application ---
class UnifiedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified HDS & APC Image Analysis Tool")
        self.root.geometry("1600x900")
        self.image = None
        self.file_path = None
        self.hds_result = None
        self.apc_result = None
        self.status_var = tk.StringVar(value="Select an image to begin.")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        # Title
        title_label = ttk.Label(main_frame, text="Unified HDS Denoising & APC Edge Detection", font=("Arial", 18, "bold"), foreground="#1a237e")
        title_label.pack(pady=(0, 5))
        subtitle_label = ttk.Label(main_frame, text="Select an image and process it using both HDS Denoising and various edge detection methods.", font=("Arial", 11, "italic"), foreground="#333366")
        subtitle_label.pack(pady=(0, 18))
        # File selection
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="📁 Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Label(file_frame, textvariable=self.status_var, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        # Edge method selection buttons above APC image section
        self.method_var = tk.StringVar(value="APC")
        self.method_names = ["APC", "Sobel", "Prewitt", "Roberts", "Scharr", "CentralDiff", "Canny", "LOG"]
        self.method_buttons_frame = None  # Will be created below APC panel
        # Process button
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=5)
        self.process_button = ttk.Button(process_frame, text="🔄 Process Image (HDS + APC)", command=self.process_image, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=5)
        # Results panels
        results_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        results_paned.pack(fill=tk.BOTH, expand=True, pady=10)
        # Original image panel
        self.original_panel = ttk.LabelFrame(results_paned, text="Original Image")
        self.original_label = ttk.Label(self.original_panel)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        self.save_orig_btn = ttk.Button(self.original_panel, text="Save Original", command=self.save_original, state="disabled")
        self.save_orig_btn.pack(pady=5)
        results_paned.add(self.original_panel, width=400)
        # HDS Denoised panel
        self.hds_panel = ttk.LabelFrame(results_paned, text="HDS Denoised Image")
        self.hds_label = ttk.Label(self.hds_panel)
        self.hds_label.pack(fill=tk.BOTH, expand=True)
        self.save_hds_btn = ttk.Button(self.hds_panel, text="Save HDS Result", command=self.save_hds, state="disabled")
        self.save_hds_btn.pack(pady=5)
        results_paned.add(self.hds_panel, width=400)
        # APC Edge panel
        self.apc_panel = ttk.LabelFrame(results_paned, text="APC Edge Map")
        # Add method selection buttons above APC image section
        self.method_buttons_frame = ttk.Frame(self.apc_panel)
        self.method_buttons_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        for method in self.method_names:
            btn = ttk.Button(self.method_buttons_frame, text=method, width=10,
                            command=lambda m=method: self.set_method_and_process(m))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.apc_label = ttk.Label(self.apc_panel)
        self.apc_label.pack(fill=tk.BOTH, expand=True)
        self.save_apc_btn = ttk.Button(self.apc_panel, text="Save APC Result", command=self.save_apc, state="disabled")
        self.save_apc_btn.pack(pady=5)
        results_paned.add(self.apc_panel, width=400)

    def set_method_and_process(self, method):
        self.method_var.set(method)
        self.process_image()

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.file_path = file_path
            img = cv2.imread(file_path)
            if img is None:
                self.status_var.set("Failed to load image.")
                self.process_button.config(state="disabled")
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image = img_rgb
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            self.display_image(self.image, self.original_label)
            self.process_button.config(state="normal")
            self.save_orig_btn.config(state="normal")
            self.hds_label.config(image="", text="")
            self.apc_label.config(image="", text="")
            self.save_hds_btn.config(state="disabled")
            self.save_apc_btn.config(state="disabled")

    def process_image(self):
        if self.image is None:
            self.status_var.set("No image loaded.")
            return
        self.status_var.set(f"Processing with HDS and {self.method_var.get()}...")
        # HDS Denoising (on grayscale)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_float = gray.astype(np.float64) / 255.0
        hds_result = hybrid_denoise(img_float, img_float.copy(), iterations=200)
        hds_img = cv2.cvtColor((hds_result * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.hds_result = hds_img
        self.display_image(hds_img, self.hds_label)
        self.save_hds_btn.config(state="normal")
        # APC Edge Detection (selected method)
        detector = EdgeDetection()
        method = self.method_var.get()
        if method == "APC":
            apc_map = detector.adaptive_principal_curvature(img_float, sigma=1.0, alpha=0.5)
        elif method == "Sobel":
            apc_map = sobel(img_float)
        elif method == "Prewitt":
            apc_map = prewitt(img_float)
        elif method == "Roberts":
            apc_map = roberts(img_float)
        elif method == "Scharr":
            apc_map = scharr(img_float)
        elif method == "CentralDiff":
            apc_map = detector.central_difference(img_float)
        elif method == "Canny":
            apc_map = detector.canny_edge_detection(img_float, low_threshold=0.1, high_threshold=0.2)
        elif method == "LOG":
            apc_map = detector.laplacian_of_gaussian(img_float, sigma=1.0)
        else:
            apc_map = detector.adaptive_principal_curvature(img_float, sigma=1.0, alpha=0.5)
        apc_img = (apc_map * 255).astype(np.uint8)
        apc_img_color = cv2.applyColorMap(apc_img, cv2.COLORMAP_JET)
        apc_img_color = cv2.cvtColor(apc_img_color, cv2.COLOR_BGR2RGB)
        self.apc_result = apc_img_color
        self.display_image(apc_img_color, self.apc_label)
        self.save_apc_btn.config(state="normal")
        self.status_var.set(f"Processing complete. Method: {method}")

    def display_image(self, img, label):
        h, w = img.shape[:2]
        max_h, max_w = 350, 350
        scale = min(max_h / h, max_w / w, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk

    def save_original(self):
        if self.image is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if save_path:
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)

    def save_hds(self):
        if self.hds_result is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if save_path:
            img_bgr = cv2.cvtColor(self.hds_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)

    def save_apc(self):
        if self.apc_result is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if save_path:
            img_bgr = cv2.cvtColor(self.apc_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedApp(root)
    root.mainloop()
