import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog, messagebox, Frame, StringVar, IntVar, DoubleVar
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
import os
from datetime import datetime
import threading
import queue
import time

# --- Configuration Parameters (from paper) ---
LAMBDA = 0.05
DT = 0.15  # Time step (tau)
G_PARAM = 1.3  # This likely acts as the alpha*beta factor in the paper
H_PARAM = 1.0  # Shape-defining constant for PM
EPSILON = 1e-8 # Small stabilizing constant for numerical stability

# --- Denoising Function Definitions (KEPT AS ORIGINAL) ---

def compute_gradients_and_coefficients(image, H_param, G_param, epsilon):
    """
    Computes first-order finite differences (gradients) and conduction coefficients.
    Uses padding to handle boundary conditions (Neumann).
    """

    # Pad the image to handle boundaries using 'edge' mode for Neumann conditions
    # This padding needs to be applied BEFORE calculating gradients from neighbors.
    # The image passed here `u` is already of original size (rows, cols)
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    # Extract slices for neighbors from the padded image
    # u_ij is the center pixel (padded_image[1:-1, 1:-1])
    # u_N is padded_image[:-2, 1:-1]
    # u_S is padded_image[2:, 1:-1]
    # u_W is padded_image[1:-1, :-2]
    # u_E is padded_image[1:-1, 2:]

    # Gradients as defined in the paper's section 2.2 for u_ij (current pixel):
    # ∇u_jiW = u_ji − u_j−1,i => u[i,j] - u[i,j-1]
    grad_u_W = image - padded_image[1:-1, :-2]
    # ∇u_jiN = u_ji − u_j,i−1 => u[i,j] - u[i-1,j]
    grad_u_N = image - padded_image[:-2, 1:-1]
    # ∇u_jiS = u_j+1,i − u_ji => u[i+1,j] - u[i,j]
    grad_u_S = padded_image[2:, 1:-1] - image
    # ∇u_jiE = u_j,i+1 − u_ji => u[i,j+1] - u[i,j]
    grad_u_E = padded_image[1:-1, 2:] - image

    # Compute conduction coefficients for all pixels simultaneously
    # c = G_PARAM * (1 / (1 + |grad|^2 / H_PARAM^2) + 1 / (|grad| + epsilon))

    c_N = G_PARAM * (1.0 / (1.0 + (grad_u_N**2) / (H_PARAM**2)) + 1.0 / (np.abs(grad_u_N) + epsilon))
    c_S = G_PARAM * (1.0 / (1.0 + (grad_u_S**2) / (H_PARAM**2)) + 1.0 / (np.abs(grad_u_S) + epsilon))
    c_W = G_PARAM * (1.0 / (1.0 + (grad_u_W**2) / (H_PARAM**2)) + 1.0 / (np.abs(grad_u_W) + epsilon))
    c_E = G_PARAM * (1.0 / (1.0 + (grad_u_E**2) / (H_PARAM**2)) + 1.0 / (np.abs(grad_u_E) + epsilon))

    return grad_u_W, grad_u_N, grad_u_S, grad_u_E, c_N, c_S, c_W, c_E

def hybrid_denoise(noisy_image_float, original_f_float, iterations=200, progress_callback=None):
    """
    Applies the hybrid diffusion-steered denoising model to the image.
    progress_callback: function to call with (current_iteration, total_iterations)
    """
    u = noisy_image_float.copy() # Current denoised image
    f = original_f_float.copy() # Original noisy image (for log-prior term)

    for n in range(iterations):
        # Update progress every 10 iterations
        if progress_callback and n % 10 == 0:
            progress_callback(n, iterations)

        # Compute gradients and conduction coefficients for the current image 'u'
        grad_u_W, grad_u_N, grad_u_S, grad_u_E, c_N, c_S, c_W, c_E = \
            compute_gradients_and_coefficients(u, H_PARAM, G_PARAM, EPSILON)

        # --- Compute the Discrete Divergence Term (diffusion part) ---
        # The paper's Eq. 9: div(u) = cN * grad_u_N + cS * grad_u_S + cW * grad_u_W + cE * grad_u_E
        # This is a direct sum as specified. The previous attempts to apply flux differences were based on a more
        # standard PDE discretization. Let's strictly adhere to the paper's formula for the divergence.

        # Initialize div_term to zeros
        div_term = np.zeros_like(u)

        # Add contributions based on paper's Eq. 9
        div_term += c_N * grad_u_N
        div_term += c_S * grad_u_S
        div_term += c_W * grad_u_W
        div_term += c_E * grad_u_E

        # --- Log-based prior term for multiplicative noise (from Eq. 10, simplified) ---
        log_prior_term = LAMBDA * ((u / (f + EPSILON)) - 1.0) * (1.0 / (u + EPSILON))

        # --- Update u using the Steepest Descent Equation (Eq. 10) ---
        u_next = u + DT * (div_term - log_prior_term)

        # Debugging: Check for NaNs or infs
        if np.isnan(u_next).any() or np.isinf(u_next).any():
            print(f"Iteration {n}: NaN or Inf encountered in u_next. Divergence!")
            print(f"Min/Max u: {np.min(u)}, {np.max(u)}")
            print(f"Min/Max div_term: {np.min(div_term)}, {np.max(div_term)}")
            print(f"Min/Max log_prior_term: {np.min(log_prior_term)}, {np.max(log_prior_term)}")
            u = np.clip(u, 0.0, 1.0) # Clip to prevent further spread of invalid values
            break # Stop if it diverges

        # Clip values to [0, 1] range to maintain valid image intensities
        u = np.clip(u_next, 0.0, 1.0)

    # Final progress update
    if progress_callback:
        progress_callback(iterations, iterations)

    return u # Return denoised image normalized to [0, 1]

def process_color_image(img_bgr, iterations=300, progress_callback=None):
    """
    Process color images by applying denoising to each channel separately
    """
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Split into channels
    r_channel = img_rgb[:, :, 0].astype(np.float64) / 255.0
    g_channel = img_rgb[:, :, 1].astype(np.float64) / 255.0
    b_channel = img_rgb[:, :, 2].astype(np.float64) / 255.0

    def channel_progress(current, total, channel_name, channel_num):
        if progress_callback:
            # Calculate overall progress across all 3 channels
            overall_progress = (channel_num - 1) * iterations + current
            overall_total = 3 * iterations
            progress_callback(overall_progress, overall_total, f"Processing {channel_name} channel...")

    # Denoise each channel with progress tracking
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

    # Combine channels back
    denoised_rgb = np.stack([r_denoised, g_denoised, b_denoised], axis=2)
    denoised_rgb = (denoised_rgb * 255).astype(np.uint8)

    if progress_callback:
        progress_callback(3 * iterations, 3 * iterations, "Combining channels...")

    return denoised_rgb

# --- Enhanced GUI Application ---

class AdvancedDenoisingApp:
    def __init__(self, master):
        self.master = master
        master.title("Block 2 : HDS edge detection")
        master.geometry("1400x900")
        master.configure(bg='#f0f0f0')

        # Variables
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
        self.check_queue()  # Start checking for results

    def setup_ui(self):
        # Main container
        main_frame = Frame(self.master, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title_label = Label(main_frame, text="Block 2 : HDS edge detection",
                           font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333333')
        title_label.pack(pady=(0, 20))

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

        # Parameters in grid
        param_grid = Frame(param_frame, bg='#ffffff')
        param_grid.pack(fill='x', pady=5)

        # Iterations
        Label(param_grid, text="Iterations:", bg='#ffffff').grid(row=0, column=0, sticky='w', padx=(0, 5))
        iterations_scale = ttk.Scale(param_grid, from_=50, to=500, variable=self.iterations_var,
                                    orient='horizontal', length=100)
        iterations_scale.grid(row=0, column=1, sticky='ew', padx=5)
        self.iterations_label = Label(param_grid, text=str(self.iterations_var.get()), bg='#ffffff')
        self.iterations_label.grid(row=0, column=2, padx=5)

        # Lambda
        Label(param_grid, text="Lambda:", bg='#ffffff').grid(row=0, column=3, sticky='w', padx=(20, 5))
        lambda_scale = ttk.Scale(param_grid, from_=0.01, to=0.2, variable=self.lambda_var,
                                orient='horizontal', length=100)
        lambda_scale.grid(row=0, column=4, sticky='ew', padx=5)
        self.lambda_label = Label(param_grid, text=f"{self.lambda_var.get():.3f}", bg='#ffffff')
        self.lambda_label.grid(row=0, column=5, padx=5)

        # Bind scale events
        iterations_scale.configure(command=lambda v: self.iterations_label.config(text=str(int(float(v)))))
        lambda_scale.configure(command=lambda v: self.lambda_label.config(text=f"{float(v):.3f}"))

        # Status and progress
        status_frame = Frame(control_frame, bg='#ffffff')
        status_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.status_label = Label(status_frame, text="Ready to process image",
                                 font=('Arial', 10), bg='#ffffff', fg='#666666')
        self.status_label.pack(anchor='w')

        # Progress bar with percentage
        progress_frame = Frame(status_frame, bg='#ffffff')
        progress_frame.pack(fill='x', pady=5)

        self.progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress.pack(side='left', fill='x', expand=True)

        self.progress_label = Label(progress_frame, text="0%", bg='#ffffff', width=5)
        self.progress_label.pack(side='right', padx=(5, 0))

        # Cancel button (hidden by default)
        self.cancel_button = Button(progress_frame, text="Cancel",
                                   command=self.cancel_processing, bg='#f44336', fg='white',
                                   font=('Arial', 9), state='disabled')
        self.cancel_button.pack(side='right', padx=(5, 5))

        # Image display area
        display_frame = Frame(main_frame, bg='#ffffff', relief='raised', bd=2)
        display_frame.pack(fill='both', expand=True)

        # Image panels
        images_frame = Frame(display_frame, bg='#ffffff')
        images_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure grid weights
        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)
        images_frame.grid_columnconfigure(2, weight=1)
        images_frame.grid_rowconfigure(0, weight=1)

        # Original image panel
        orig_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        orig_panel.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        Label(orig_panel, text="Original Image", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.original_display = Label(orig_panel, bg='#f8f8f8')
        self.original_display.pack(expand=True)

        # Denoised image panel
        denoised_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        denoised_panel.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        Label(denoised_panel, text="Denoised Image", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.denoised_display = Label(denoised_panel, bg='#f8f8f8')
        self.denoised_display.pack(expand=True)

        # Difference image panel
        diff_panel = Frame(images_frame, bg='#f8f8f8', relief='sunken', bd=1)
        diff_panel.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        Label(diff_panel, text="Noise Removed", font=('Arial', 12, 'bold'),
              bg='#f8f8f8').pack(pady=5)
        self.diff_display = Label(diff_panel, bg='#f8f8f8')
        self.diff_display.pack(expand=True)

        # Info panel
        info_frame = Frame(display_frame, bg='#ffffff')
        info_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.info_label = Label(info_frame, text="Load an image to begin analysis",
                               font=('Arial', 10), bg='#ffffff', fg='#666666')
        self.info_label.pack(anchor='w')

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("All Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.current_file_path = file_path
        self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        try:
            self.status_label.config(text="Loading image...")
            self.master.update_idletasks()

            # Load image
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Could not open or find the image.")

            # Check if image is color or grayscale
            if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                self.is_color = True
                # Convert BGR to RGB for display
                self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                display_image = self.original_image
            else:
                self.is_color = False
                self.original_image = img_bgr
                display_image = img_bgr

            # Display original image
            self.display_image_in_label(display_image, self.original_display)

            # Update info
            height, width = self.original_image.shape[:2]
            channels = "Color (RGB)" if self.is_color else "Grayscale"
            file_size = os.path.getsize(file_path) / 1024  # KB

            info_text = f"File: {os.path.basename(file_path)} | Size: {width}x{height} | Type: {channels} | File Size: {file_size:.1f} KB"
            self.info_label.config(text=info_text)

            # Enable process button
            self.process_button.config(state='normal')
            self.status_label.config(text="Image loaded successfully. Ready to process.")

        except Exception as e:
            error_message = f"Error loading image: {str(e)}"
            messagebox.showerror("Error", error_message)
            self.status_label.config(text="Error loading image.")
            self.clear_displays()

    def process_image(self):
        if self.original_image is None:
            return

        # Prevent multiple processing threads
        if self.processing_thread and self.processing_thread.is_alive():
            return

        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self._process_image_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Update UI for processing state
        self.process_button.config(state='disabled')
        self.upload_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        self.progress['value'] = 0
        self.progress_label.config(text="0%")
        self.status_label.config(text="Initializing processing...")

    def _process_image_thread(self):
        """Background thread for image processing"""
        try:
            iterations = int(self.iterations_var.get())

            # Update global parameters
            global LAMBDA, DT, G_PARAM, H_PARAM
            LAMBDA = self.lambda_var.get()

            def progress_callback(current, total, message="Processing..."):
                # Send progress update to main thread
                progress_percent = int((current / total) * 100)
                self.result_queue.put(('progress', progress_percent, message))

            if self.is_color:
                # Process color image
                img_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                denoised_image = process_color_image(img_bgr, iterations, progress_callback)
            else:
                # Process grayscale image
                if len(self.original_image.shape) == 3:
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = self.original_image

                noisy_float = gray_image.astype(np.float64) / 255.0
                denoised_float = hybrid_denoise(noisy_float, noisy_float.copy(),
                                              iterations, progress_callback)
                denoised_image = (denoised_float * 255).astype(np.uint8)

            # Send result to main thread
            self.result_queue.put(('success', denoised_image))

        except Exception as e:
            # Send error to main thread
            self.result_queue.put(('error', str(e)))

    def cancel_processing(self):
        """Cancel the current processing (note: this is a simple implementation)"""
        self.status_label.config(text="Cancelling processing...")
        self.cancel_button.config(state='disabled')
        # Note: Actual cancellation of numpy operations is complex,
        # so this mainly prevents UI updates and resets the interface
        self.reset_processing_ui()

    def reset_processing_ui(self):
        """Reset UI to normal state after processing"""
        self.process_button.config(state='normal')
        self.upload_button.config(state='normal')
        self.cancel_button.config(state='disabled')
        self.progress['value'] = 0
        self.progress_label.config(text="0%")

    def check_queue(self):
        """Check for results from the processing thread"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                result_type = result[0]

                if result_type == 'progress':
                    progress_percent, message = result[1], result[2]
                    self.progress['value'] = progress_percent
                    self.progress_label.config(text=f"{progress_percent}%")
                    self.status_label.config(text=message)

                elif result_type == 'success':
                    self.denoised_image = result[1]
                    self.display_results()
                    self.reset_processing_ui()
                    self.save_button.config(state='normal')
                    self.status_label.config(text="Processing complete!")
                    self.progress['value'] = 100
                    self.progress_label.config(text="100%")

                elif result_type == 'error':
                    error_message = f"Error processing image: {result[1]}"
                    messagebox.showerror("Error", error_message)
                    self.reset_processing_ui()
                    self.status_label.config(text="Error during processing.")
                    print(f"Processing error: {result[1]}")

        except queue.Empty:
            pass

        # Schedule next check
        self.master.after(100, self.check_queue)

    def display_results(self):
        # Display denoised image
        self.display_image_in_label(self.denoised_image, self.denoised_display)

        # Create and display difference image
        if self.is_color:
            # For color images, compute difference and enhance
            original_for_diff = self.original_image
            diff_img = np.abs(original_for_diff.astype(np.int16) - self.denoised_image.astype(np.int16))
            diff_img = np.clip(diff_img * 3, 0, 255).astype(np.uint8)  # Enhance visibility
        else:
            # For grayscale images
            if len(self.original_image.shape) == 3:
                original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = self.original_image

            diff_img = np.abs(original_gray.astype(np.int16) - self.denoised_image.astype(np.int16))
            diff_img = np.clip(diff_img * 5, 0, 255).astype(np.uint8)  # Enhance visibility

        self.display_image_in_label(diff_img, self.diff_display)

    def display_image_in_label(self, image, label):
        # Convert to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        else:  # Color
            pil_image = Image.fromarray(image)

        # Resize for display
        label_width = 400
        label_height = 300

        # Calculate aspect ratio
        img_width, img_height = pil_image.size
        aspect_ratio = img_width / img_height

        if aspect_ratio > label_width / label_height:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)

        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(pil_image)
        label.config(image=photo)
        label.image = photo  # Keep a reference

    def save_results(self):
        if self.denoised_image is None:
            messagebox.showwarning("Warning", "No processed image to save.")
            return

        # Get save directory
        save_dir = filedialog.askdirectory(title="Select folder to save results")
        if not save_dir:
            return

        try:
            # Generate filename base
            original_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{original_name}_denoised_{timestamp}"

            # Save denoised image
            if self.is_color:
                # Save as RGB
                denoised_pil = Image.fromarray(self.denoised_image)
                denoised_path = os.path.join(save_dir, f"{base_name}.png")
                denoised_pil.save(denoised_path)
            else:
                # Save grayscale
                denoised_path = os.path.join(save_dir, f"{base_name}.png")
                cv2.imwrite(denoised_path, self.denoised_image)

            # Save original for comparison
            if self.is_color:
                original_pil = Image.fromarray(self.original_image)
                original_path = os.path.join(save_dir, f"{original_name}_original.png")
                original_pil.save(original_path)
            else:
                original_path = os.path.join(save_dir, f"{original_name}_original.png")
                if len(self.original_image.shape) == 3:
                    cv2.imwrite(original_path, self.original_image)
                else:
                    cv2.imwrite(original_path, self.original_image)

            # Create difference image and save
            if self.is_color:
                diff_img = np.abs(self.original_image.astype(np.int16) - self.denoised_image.astype(np.int16))
                diff_img = np.clip(diff_img * 3, 0, 255).astype(np.uint8)
                diff_pil = Image.fromarray(diff_img)
                diff_path = os.path.join(save_dir, f"{base_name}_noise_removed.png")
                diff_pil.save(diff_path)
            else:
                if len(self.original_image.shape) == 3:
                    original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    original_gray = self.original_image
                diff_img = np.abs(original_gray.astype(np.int16) - self.denoised_image.astype(np.int16))
                diff_img = np.clip(diff_img * 5, 0, 255).astype(np.uint8)
                diff_path = os.path.join(save_dir, f"{base_name}_noise_removed.png")
                cv2.imwrite(diff_path, diff_img)

            # Save processing parameters
            params_path = os.path.join(save_dir, f"{base_name}_parameters.txt")
            with open(params_path, 'w') as f:
                f.write(f"Image Denoising Parameters\n")
                f.write(f"========================\n")
                f.write(f"Original file: {self.current_file_path}\n")
                f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image type: {'Color' if self.is_color else 'Grayscale'}\n")
                f.write(f"Iterations: {self.iterations_var.get()}\n")
                f.write(f"Lambda: {self.lambda_var.get()}\n")
                f.write(f"DT: {DT}\n")
                f.write(f"G_PARAM: {G_PARAM}\n")
                f.write(f"H_PARAM: {H_PARAM}\n")
                f.write(f"\nOutput files:\n")
                f.write(f"- Denoised image: {os.path.basename(denoised_path)}\n")
                f.write(f"- Original image: {os.path.basename(original_path)}\n")
                f.write(f"- Noise removed: {os.path.basename(diff_path)}\n")

            messagebox.showinfo("Success",
                              f"Results saved successfully!\n\nFiles saved:\n"
                              f"• Denoised image\n"
                              f"• Original image\n"
                              f"• Noise visualization\n"
                              f"• Processing parameters\n\n"
                              f"Location: {save_dir}")

            self.status_label.config(text="Results saved successfully!")

        except Exception as e:
            error_message = f"Error saving files: {str(e)}"
            messagebox.showerror("Error", error_message)
            self.status_label.config(text="Error saving files.")

    def clear_displays(self):
        self.original_display.config(image=None)
        self.denoised_display.config(image=None)
        self.diff_display.config(image=None)
        self.original_image = None
        self.denoised_image = None
        self.process_button.config(state='disabled')
        self.save_button.config(state='disabled')

if __name__ == "__main__":
    root = Tk()
    app = AdvancedDenoisingApp(root)
    root.mainloop()