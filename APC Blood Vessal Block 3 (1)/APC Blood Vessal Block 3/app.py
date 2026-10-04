import os
import sys
import traceback
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sobel, prewitt, roberts, scharr, laplace
from skimage.morphology import remove_small_objects, binary_closing, disk

# Try importing tkinter with error handling
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TKINTER_AVAILABLE = True
except ImportError:
    print("Error: tkinter is not available. Will use command line mode instead.")
    TKINTER_AVAILABLE = False

# Try importing matplotlib with error handling
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Error: matplotlib is not available. Will use OpenCV for display instead.")
    MATPLOTLIB_AVAILABLE = False


class EdgeDetection:
    """
    A class for detecting edges in images using various methods.
    """

    def __init__(self):
        """Initialize the EdgeDetection class"""
        pass

    def preprocess_image(self, image):
        """
        Preprocess the input image (convert to grayscale and normalize)
        """
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32) / 255.0
        return gray

    def adaptive_principal_curvature(self, image, sigma=1.0, alpha=0.5):
        """
        Calculate adaptive principal curvature for edge enhancement
        """
        # Compute Hessian matrix
        hessian = hessian_matrix(image, sigma=sigma, order='rc', use_gaussian_derivatives=False)

        # Compute eigenvalues of Hessian matrix
        eigvals = hessian_matrix_eigvals(hessian)

        # Extract maximum and minimum eigenvalues (principal curvatures)
        lambda1 = eigvals[0]  # minimum eigenvalue
        lambda2 = eigvals[1]  # maximum eigenvalue

        # Adaptive principal curvature combines both curvatures
        apc = alpha * np.abs(lambda2) + (1 - alpha) * np.abs(lambda1)

        # Normalize to 0-1 range
        apc = (apc - np.min(apc)) / (np.max(apc) - np.min(apc) + 1e-10)

        return apc

    def central_difference(self, image):
        """
        Calculate central difference derivatives for edge detection
        """
        dx = np.zeros_like(image)
        dy = np.zeros_like(image)

        # Central difference for dx
        dx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2

        # Central difference for dy
        dy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2

        # Compute gradient magnitude
        grad_mag = np.sqrt(dx**2 + dy**2)

        # Normalize
        grad_mag = (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag) + 1e-10)

        return grad_mag

    def canny_edge_detection(self, image, low_threshold=0.1, high_threshold=0.2):
        """
        Apply Canny edge detection
        """
        # Convert to uint8 for Canny
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply Canny edge detection
        edges = cv2.Canny(image_uint8,
                         int(low_threshold * 255),
                         int(high_threshold * 255))

        # Normalize to 0-1 range
        edges = edges.astype(np.float32) / 255.0

        return edges

    def laplacian_of_gaussian(self, image, sigma=1.0):
        """
        Apply Laplacian of Gaussian edge detection
        """
        # First apply Gaussian smoothing
        smoothed = gaussian_filter(image, sigma=sigma)

        # Then apply Laplacian
        log_response = laplace(smoothed)

        # Take absolute value and normalize
        log_response = np.abs(log_response)
        log_response = (log_response - np.min(log_response)) / (np.max(log_response) - np.min(log_response) + 1e-10)

        return log_response

    def detect_edges(self, image, method="Sobel", sigma=1.0, alpha=0.5, threshold=0.3,
                    post_process=True, min_size=20, closing_radius=2,
                    canny_low=0.1, canny_high=0.2):
        """
        Detect edges using the specified method
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Apply Gaussian smoothing to reduce noise (except for Canny which has its own smoothing)
        if method.upper() != "CANNY":
            smoothed = gaussian_filter(preprocessed, sigma=sigma)
        else:
            smoothed = preprocessed

        # Apply selected method
        if method.upper() == "APC":
            response = self.adaptive_principal_curvature(smoothed, sigma=sigma, alpha=alpha)
        elif method.upper() == "SOBEL":
            response = sobel(smoothed)
            response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        elif method.upper() == "PREWITT":
            response = prewitt(smoothed)
            response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        elif method.upper() == "ROBERTS":
            response = roberts(smoothed)
            response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        elif method.upper() == "SCHARR":
            response = scharr(smoothed)
            response = (response - np.min(response)) / (np.max(response) - np.min(response) + 1e-10)
        elif method.upper() == "CENTRALDIFF":
            response = self.central_difference(smoothed)
        elif method.upper() == "CANNY":
            response = self.canny_edge_detection(smoothed, canny_low, canny_high)
        elif method.upper() == "LOG":
            response = self.laplacian_of_gaussian(smoothed, sigma=sigma)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'Sobel', 'Prewitt', 'Roberts', 'Scharr', 'CentralDiff', 'Canny', 'LOG', or 'APC'.")

        # Threshold to get binary edge map
        binary = response > threshold

        # Post-processing: remove small objects and fill holes
        if post_process and method.upper() != "CANNY":  # Canny already produces clean edges
            binary = remove_small_objects(binary, min_size=min_size)
            binary = binary_closing(binary, disk(closing_radius))

        return binary, response

    def create_focused_edge_views(self, image, binary_mask, response):
        """
        Create enhanced views of the detected edges:
        1. Edge overlay (original with colored edge overlay)
        2. Edge map (binary edge map)
        3. Response map (gradient magnitude/response visualization)
        4. Enhanced edges (improved contrast in edge areas)
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            rgb_image = np.stack([image] * 3, axis=2)
        else:
            rgb_image = image.copy()

        # 1. Edge overlay (bright cyan overlay on original)
        edge_overlay = rgb_image.copy()
        if len(edge_overlay.shape) == 3:
            # Create a cyan highlight for edges
            edge_overlay[binary_mask, 0] = np.minimum(255, edge_overlay[binary_mask, 0] * 0.5)  # Reduce red
            edge_overlay[binary_mask, 1] = np.minimum(255, edge_overlay[binary_mask, 1] * 0.5 + 128)  # Increase green
            edge_overlay[binary_mask, 2] = np.minimum(255, edge_overlay[binary_mask, 2] * 0.5 + 128)  # Increase blue

        # 2. Edge map (binary visualization)
        edge_map = np.zeros_like(rgb_image)
        edge_map[binary_mask] = [255, 255, 255]  # White edges on black background

        # 3. Response map (gradient magnitude visualization with color mapping)
        response_normalized = (response * 255).astype(np.uint8)
        response_colored = cv2.applyColorMap(response_normalized, cv2.COLORMAP_JET)
        response_colored = cv2.cvtColor(response_colored, cv2.COLOR_BGR2RGB)

        # 4. Enhanced edges (show only edge areas from original with enhanced contrast)
        enhanced = np.zeros_like(rgb_image)
        enhanced[binary_mask] = rgb_image[binary_mask]

        # Enhance contrast in edge areas
        if len(enhanced.shape) == 3 and np.any(binary_mask):
            for c in range(3):
                edge_pixels = enhanced[binary_mask, c]
                if len(edge_pixels) > 0:
                    p_min, p_max = np.min(edge_pixels), np.max(edge_pixels)
                    if p_max > p_min:
                        enhanced[binary_mask, c] = np.clip(
                            255 * (enhanced[binary_mask, c] - p_min) / (p_max - p_min),
                            0, 255
                        ).astype(np.uint8)

        return edge_overlay, edge_map, response_colored, enhanced


def load_image(file_path):
    """Load an image from file."""
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")

        # Convert to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


def save_image(file_path, image):
    """Save an image to file."""
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(file_path, image)
        return success
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False


class SimpleEdgeDetectionGUI:
    """Simple GUI for edge detection"""

    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Edge Detection Tool")
        self.root.geometry("1200x700")  # Increased width to accommodate side-by-side display

        # Initialize edge detection object
        self.detector = EdgeDetection()

        # Initialize variables
        self.image = None
        self.file_path = None
        self.binary_mask = None
        self.response = None
        self.edge_overlay = None
        self.edge_map = None
        self.response_colored = None
        self.enhanced = None
        self.current_method = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Edge Detection Controls")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # File operation buttons frame
        file_buttons_frame = ttk.Frame(control_frame)
        file_buttons_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Load and Save buttons
        ttk.Button(file_buttons_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)

        # Method buttons frame
        method_frame = ttk.LabelFrame(control_frame, text="Edge Detection Methods")
        method_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create method buttons
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

        # Create buttons in two rows
        for i, (method, tooltip) in enumerate(methods_info):
            row = i // 4
            col = i % 4
            btn = ttk.Button(method_frame, text=method,
                           command=lambda m=method: self.apply_method(m),
                           width=12)
            btn.grid(row=row, column=col, padx=3, pady=3, sticky=tk.W+tk.E)

            # Add tooltip (simple status bar update on hover)
            btn.bind("<Enter>", lambda e, tip=tooltip: self.status_var.set(tip))
            btn.bind("<Leave>", lambda e: self.status_var.set(self.current_status))

        # Configure grid weights for even spacing
        for i in range(4):
            method_frame.columnconfigure(i, weight=1)

        # Parameters frame
        params_frame = ttk.LabelFrame(control_frame, text="Parameters")
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Current method display
        ttk.Label(params_frame, text="Current Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.current_method_var = tk.StringVar(value="None")
        ttk.Label(params_frame, textvariable=self.current_method_var,
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Threshold
        ttk.Label(params_frame, text="Threshold:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_entry = ttk.Entry(params_frame, textvariable=self.threshold_var, width=5)
        threshold_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        threshold_entry.bind("<Return>", lambda e: self.update_current_method())

        # Sigma (for Gaussian smoothing)
        ttk.Label(params_frame, text="Sigma:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.sigma_var = tk.DoubleVar(value=1.0)
        sigma_entry = ttk.Entry(params_frame, textvariable=self.sigma_var, width=5)
        sigma_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        sigma_entry.bind("<Return>", lambda e: self.update_current_method())

        # View selection
        ttk.Label(params_frame, text="View:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.view_var = tk.StringVar(value="original")
        view_combo = ttk.Combobox(params_frame, textvariable=self.view_var,
                                values=["original", "edge_overlay", "edge_map", "response", "enhanced"],
                                width=12)
        view_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        view_combo.bind("<<ComboboxSelected>>", self.update_view)

        # Update button
        ttk.Button(params_frame, text="Update", command=self.update_current_method).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load an image to start")
        self.current_status = "Ready - Load an image to start"
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Image frame - now using a PanedWindow to show original and processed side by side
        self.image_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.image_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Original image frame
        self.original_frame = ttk.LabelFrame(self.image_paned, text="Original Image")
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        self.image_paned.add(self.original_frame, width=400)

        # Processed image frame
        self.processed_frame = ttk.LabelFrame(self.image_paned, text="Processed Image")
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True)
        self.image_paned.add(self.processed_frame, width=400)

    def load_image(self):
        """Load an image from file"""
        try:
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
                    # Display original image in the left panel
                    self.display_image(self.image, panel="original")
                    # Reset processed results
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
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.current_status = f"Error: {str(e)}"
            if messagebox:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def apply_method(self, method):
        """Apply the selected edge detection method and show response immediately"""
        try:
            if self.image is None:
                self.status_var.set("Please load an image first")
                self.current_status = "Please load an image first"
                return

            # Update current method
            self.current_method = method
            self.current_method_var.set(method)

            # Get parameters
            threshold = float(self.threshold_var.get())
            sigma = float(self.sigma_var.get())

            # Set reasonable defaults for other parameters
            alpha = 0.5
            post_process = True
            min_size = 20
            closing_radius = 2
            canny_low = 0.1
            canny_high = 0.2

            # Process image with the selected method
            self.binary_mask, self.response = self.detector.detect_edges(
                self.image, method=method, sigma=sigma, alpha=alpha,
                threshold=threshold, post_process=post_process,
                min_size=min_size, closing_radius=closing_radius,
                canny_low=canny_low, canny_high=canny_high
            )

            # Create enhanced views
            self.edge_overlay, self.edge_map, self.response_colored, self.enhanced = self.detector.create_focused_edge_views(
                self.image, self.binary_mask, self.response
            )

            # Automatically show the response view
            self.view_var.set("response")
            self.update_view()

            status_msg = f"{method} applied - Threshold: {threshold:.2f}, Sigma: {sigma:.1f}"
            self.status_var.set(status_msg)
            self.current_status = status_msg

        except Exception as e:
            error_msg = f"Error applying {method}: {str(e)}"
            self.status_var.set(error_msg)
            self.current_status = error_msg
            traceback.print_exc()
            if messagebox:
                messagebox.showerror("Error", error_msg)

    def update_current_method(self):
        """Update the current method with new parameters"""
        if self.current_method and self.image is not None:
            self.apply_method(self.current_method)
        else:
            self.status_var.set("No method selected or no image loaded")
            self.current_status = "No method selected or no image loaded"

    def process_image(self):
        """Legacy method - now redirects to apply_method"""
        if self.current_method:
            self.apply_method(self.current_method)
        else:
            self.status_var.set("Please select a method by clicking one of the method buttons")
            self.current_status = "Please select a method by clicking one of the method buttons"

    def update_view(self, event=None):
        """Update the displayed image based on view selection"""
        try:
            if self.image is None:
                return

            view = self.view_var.get()

            # Always keep original image displayed in left panel
            self.display_image(self.image, panel="original")

            # Update right panel based on selection
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
        except Exception as e:
            self.status_var.set(f"Error updating view: {str(e)}")
            traceback.print_exc()

    def display_image(self, img, panel="original"):
        """Display an image on the specified panel (original or processed)"""
        try:
            h, w = img.shape[:2]
            max_h = 550
            max_w = 400  # Reduced to fit side by side

            # Scale down if needed
            if h > max_h or w > max_w:
                scale = min(max_h / h, max_w / w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
            else:
                img_resized = img.copy()

            # Convert to format for tkinter
            img_display = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            img_display = cv2.imencode('.png', img_display)[1].tobytes()
            img_tk = tk.PhotoImage(data=img_display)

            # Update appropriate label
            if panel == "original":
                self.original_label.configure(image=img_tk)
                self.original_label.image = img_tk  # Keep a reference
            else:
                self.processed_label.configure(image=img_tk)
                self.processed_label.image = img_tk  # Keep a reference
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")
            traceback.print_exc()

    def save_results(self):
        """Save results to files"""
        try:
            if self.image is None or self.binary_mask is None:
                self.status_var.set("Process an image before saving")
                self.current_status = "Process an image before saving"
                return

            # Ask for directory
            save_dir = filedialog.askdirectory(title="Select Directory to Save Results")

            if not save_dir:
                return

            # Base filename
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            method = self.current_method or "unknown"

            # Save original image
            original_path = os.path.join(save_dir, f"{base_filename}_original.png")
            save_image(original_path, self.image)

            # Save edge overlay
            if self.edge_overlay is not None:
                overlay_path = os.path.join(save_dir, f"{base_filename}_{method}_overlay.png")
                save_image(overlay_path, self.edge_overlay)

            # Save edge map
            if self.edge_map is not None:
                map_path = os.path.join(save_dir, f"{base_filename}_{method}_edges.png")
                save_image(map_path, self.edge_map)

            # Save response visualization
            if self.response_colored is not None:
                response_path = os.path.join(save_dir, f"{base_filename}_{method}_response.png")
                save_image(response_path, self.response_colored)

            # Save enhanced view
            if self.enhanced is not None:
                enhanced_path = os.path.join(save_dir, f"{base_filename}_{method}_enhanced.png")
                save_image(enhanced_path, self.enhanced)

            status_msg = f"Results saved to {save_dir}"
            self.status_var.set(status_msg)
            self.current_status = status_msg
        except Exception as e:
            error_msg = f"Error saving: {str(e)}"
            self.status_var.set(error_msg)
            self.current_status = error_msg
            traceback.print_exc()
            if messagebox:
                messagebox.showerror("Error", error_msg)


def command_line_mode():
    """Run in command line mode when GUI is not available"""
    print("\n=== Edge Detection Tool (Command Line Mode) ===\n")

    # Ask for image path
    image_path = input("Enter path to the image file: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    try:
        # Load image
        print("Loading image...")
        image = load_image(image_path)
        if image is None:
            print("Failed to load image.")
            return

        # Ask for method
        print("\nAvailable edge detection methods:")
        print("1. Sobel")
        print("2. Prewitt")
        print("3. Roberts")
        print("4. Scharr")
        print("5. CentralDiff (Central Difference)")
        print("6. Canny")
        print("7. LOG (Laplacian of Gaussian)")
        print("8. APC (Adaptive Principal Curvature)")

        method_choice = input("Select method (1-8, default=1): ").strip()
        methods = ["Sobel", "Prewitt", "Roberts", "Scharr", "CentralDiff", "Canny", "LOG", "APC"]
        method = methods[0]  # Default

        if method_choice.isdigit() and 1 <= int(method_choice) <= 8:
            method = methods[int(method_choice) - 1]

        # Ask for threshold
        threshold_input = input("Enter threshold (0.0-1.0, default=0.3): ").strip()
        threshold = 0.3  # Default

        if threshold_input:
            try:
                threshold = float(threshold_input)
                threshold = max(0.0, min(1.0, threshold))  # Clamp to valid range
            except ValueError:
                print("Invalid threshold. Using default value of 0.3.")

        # Ask for sigma
        sigma_input = input("Enter sigma for Gaussian smoothing (default=1.0): ").strip()
        sigma = 1.0  # Default

        if sigma_input:
            try:
                sigma = float(sigma_input)
                sigma = max(0.1, sigma)  # Ensure positive
            except ValueError:
                print("Invalid sigma. Using default value of 1.0.")

        # Process image
        print(f"\nProcessing image with {method}, threshold={threshold}, sigma={sigma}...")
        detector = EdgeDetection()
        binary_mask, response = detector.detect_edges(
            image, method=method, threshold=threshold, sigma=sigma
        )

        # Create enhanced views
        edge_overlay, edge_map, response_colored, enhanced = detector.create_focused_edge_views(
            image, binary_mask, response
        )

        # Ask for save directory
        save_dir = input("\nEnter directory to save results (default=current directory): ").strip()
        if not save_dir:
            save_dir = os.getcwd()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Base filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save results
        print("Saving results...")

        # Save original image
        original_path = os.path.join(save_dir, f"{base_filename}_original.png")
        save_image(original_path, image)
        print(f"- Original image saved to: {original_path}")

        # Save edge overlay
        overlay_path = os.path.join(save_dir, f"{base_filename}_{method}_overlay.png")
        save_image(overlay_path, edge_overlay)
        print(f"- Edge overlay saved to: {overlay_path}")

        # Save edge map
        map_path = os.path.join(save_dir, f"{base_filename}_{method}_edges.png")
        save_image(map_path, edge_map)
        print(f"- Edge map saved to: {map_path}")

        # Save response visualization
        response_path = os.path.join(save_dir, f"{base_filename}_{method}_response.png")
        save_image(response_path, response_colored)
        print(f"- Response visualization saved to: {response_path}")

        # Save enhanced view
        enhanced_path = os.path.join(save_dir, f"{base_filename}_{method}_enhanced.png")
        save_image(enhanced_path, enhanced)
        print(f"- Enhanced edges saved to: {enhanced_path}")

        print(f"\nEdge detection complete! Method used: {method}")
        print(f"Check the response visualization to see the gradient magnitude/response map.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()


# Main function
def main():
    """Main function to run the application"""
    try:
        # First try GUI mode
        if TKINTER_AVAILABLE:
            try:
                root = tk.Tk()
                app = SimpleEdgeDetectionGUI(root)
                root.mainloop()
                return
            except Exception as e:
                print(f"Error initializing GUI: {str(e)}")
                traceback.print_exc()
                print("\nFalling back to command line mode...\n")

        # Fall back to command line mode
        command_line_mode()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        print("\nApplication terminated due to an error.")


# Run the application
if __name__ == "__main__":
    print("Starting Edge Detection Tool...")

    # Check for required dependencies
    missing_deps = []

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import cv2
    except ImportError:
        missing_deps.append("cv2 (OpenCV)")

    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")

    try:
        import skimage
    except ImportError:
        missing_deps.append("scikit-image")

    if missing_deps:
        print("\nERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies with:")
        print("pip install numpy opencv-python scipy scikit-image")
        sys.exit(1)

    main()