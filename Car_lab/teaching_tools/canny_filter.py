import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import sys
import os


class CannyDemo:
    def __init__(self, image_path):
        self.root = tk.Tk()
        self.root.title("Canny Edge Detection Demo")
        self.root.geometry("1200x1000")
        
        # Load and validate image
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            sys.exit(1)
            
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Unable to load image: {image_path}")
            sys.exit(1)
            
        # Convert to grayscale for processing
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Resize image if too large for display
        self.display_width = 400
        self.display_height = 300
        self.gray_resized = cv2.resize(self.gray_image, (self.display_width, self.display_height))
        
        # Initialize parameters
        self.low_threshold = tk.IntVar(value=50)
        self.high_threshold = tk.IntVar(value=150)
        self.blur_enabled = tk.BooleanVar(value=False)
        self.blur_kernel = tk.IntVar(value=5)
        
        # Matrix window parameters
        self.matrix_size = 12
        self.window_x = tk.IntVar(value=50)  # Top-left x position
        self.window_y = tk.IntVar(value=50)  # Top-left y position
        
        # Set slider ranges based on image size
        max_x = max(0, self.display_width - self.matrix_size)
        max_y = max(0, self.display_height - self.matrix_size)
        self.max_window_x = max_x
        self.max_window_y = max_y
        
        # Section expansion states
        self.original_expanded = tk.BooleanVar(value=True)
        self.gradient_expanded = tk.BooleanVar(value=True)
        self.canny_expanded = tk.BooleanVar(value=True)
        
        # Left panel image section expansion states
        self.original_image_expanded = tk.BooleanVar(value=True)
        self.processed_image_expanded = tk.BooleanVar(value=True)
        self.canny_image_expanded = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        # Main frame with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        
        # Left panel for images and controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))
        left_panel.columnconfigure(0, weight=1)
        
        # Original image section
        self.original_image_button = tk.Button(left_panel,
                                              text="▼ Original Image (Grayscale)",
                                              font=('Arial', 12, 'bold'),
                                              command=self.toggle_original_image_section,
                                              relief='flat', anchor='w')
        self.original_image_button.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Original image display
        self.original_image_frame = ttk.Frame(left_panel)
        self.original_image_frame.grid(row=1, column=0, pady=(0, 15))
        
        self.original_label = ttk.Label(self.original_image_frame)
        self.original_label.grid(row=0, column=0)
        
        # Processed image section
        self.processed_image_button = tk.Button(left_panel,
                                               text="▼ Processed Image (with/without Blur)",
                                               font=('Arial', 12, 'bold'),
                                               command=self.toggle_processed_image_section,
                                               relief='flat', anchor='w')
        self.processed_image_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Processed image display
        self.processed_image_frame = ttk.Frame(left_panel)
        self.processed_image_frame.grid(row=3, column=0, pady=(0, 15))
        
        self.processed_label = ttk.Label(self.processed_image_frame)
        self.processed_label.grid(row=0, column=0)
        
        # Canny image section
        self.canny_image_button = tk.Button(left_panel,
                                           text="▼ Canny Edge Detection",
                                           font=('Arial', 12, 'bold'),
                                           command=self.toggle_canny_image_section,
                                           relief='flat', anchor='w')
        self.canny_image_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Canny image display
        self.canny_image_frame = ttk.Frame(left_panel)
        self.canny_image_frame.grid(row=5, column=0, pady=(0, 20))
        
        self.canny_label = ttk.Label(self.canny_image_frame)
        self.canny_label.grid(row=0, column=0)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_panel, text="Canny Parameters", padding="10")
        controls_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        
        # Low threshold slider
        ttk.Label(controls_frame, text="Low Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.low_scale = ttk.Scale(controls_frame, from_=0, to=255, 
                                  variable=self.low_threshold, orient=tk.HORIZONTAL,
                                  command=self.on_parameter_change)
        self.low_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.low_value_label = ttk.Label(controls_frame, text="50")
        self.low_value_label.grid(row=0, column=2, sticky=tk.W)
        
        # High threshold slider
        ttk.Label(controls_frame, text="High Threshold:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.high_scale = ttk.Scale(controls_frame, from_=0, to=255, 
                                   variable=self.high_threshold, orient=tk.HORIZONTAL,
                                   command=self.on_parameter_change)
        self.high_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.high_value_label = ttk.Label(controls_frame, text="150")
        self.high_value_label.grid(row=1, column=2, sticky=tk.W)
        
        # Gaussian blur checkbox
        self.blur_checkbox = ttk.Checkbutton(controls_frame, text="Gaussian Blur", 
                                           variable=self.blur_enabled,
                                           command=self.on_parameter_change)
        self.blur_checkbox.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # Blur kernel slider
        ttk.Label(controls_frame, text="Blur Kernel Size:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        self.kernel_scale = ttk.Scale(controls_frame, from_=1, to=15, 
                                     variable=self.blur_kernel, orient=tk.HORIZONTAL,
                                     command=self.on_kernel_change)
        self.kernel_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.kernel_value_label = ttk.Label(controls_frame, text="5")
        self.kernel_value_label.grid(row=3, column=2, sticky=tk.W)
        
        # Initially disable kernel slider
        self.update_kernel_state()
        
        # Right panel for matrix display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Matrix panel title
        ttk.Label(right_panel, text="Pixel Value Analysis (12x12)", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        # Window position controls
        position_frame = ttk.LabelFrame(right_panel, text="Window Position", padding="5")
        position_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(position_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.x_scale = ttk.Scale(position_frame, from_=0, to=self.max_window_x,
                                variable=self.window_x, orient=tk.HORIZONTAL,
                                command=self.on_window_change)
        self.x_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.x_value_label = ttk.Label(position_frame, text="50")
        self.x_value_label.grid(row=0, column=2)
        
        ttk.Label(position_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.y_scale = ttk.Scale(position_frame, from_=0, to=self.max_window_y,
                                variable=self.window_y, orient=tk.HORIZONTAL,
                                command=self.on_window_change)
        self.y_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.y_value_label = ttk.Label(position_frame, text="50")
        self.y_value_label.grid(row=1, column=2)
        
        position_frame.columnconfigure(1, weight=1)
        
        # Create expandable sections
        self.create_expandable_sections(right_panel)
        
    def create_expandable_sections(self, parent):
        current_row = 2
        
        # Original Values Section
        self.original_section_button = tk.Button(parent, 
                                                text="▼ Original Grayscale Values",
                                                font=('Arial', 10, 'bold'),
                                                command=self.toggle_original_section,
                                                relief='flat', anchor='w')
        self.original_section_button.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.original_section_frame = ttk.Frame(parent)
        self.original_section_frame.grid(row=current_row+1, column=0, pady=(5, 10))
        current_row += 2
        
        # Gradient Calculations Section
        self.gradient_section_button = tk.Button(parent,
                                                text="▼ Gradient Calculations", 
                                                font=('Arial', 10, 'bold'),
                                                command=self.toggle_gradient_section,
                                                relief='flat', anchor='w')
        self.gradient_section_button.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        self.gradient_section_frame = ttk.Frame(parent)
        self.gradient_section_frame.grid(row=current_row+1, column=0, pady=(5, 10))
        current_row += 2
        
        # Final Edge Detection Section
        self.canny_section_button = tk.Button(parent,
                                             text="▼ Final Edge Detection",
                                             font=('Arial', 10, 'bold'), 
                                             command=self.toggle_canny_section,
                                             relief='flat', anchor='w')
        self.canny_section_button.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        self.canny_section_frame = ttk.Frame(parent)
        self.canny_section_frame.grid(row=current_row+1, column=0, pady=(5, 10))
        
        # Create matrix grids
        self.create_matrix_grids()
        
    def create_matrix_grids(self):
        # Create original values matrix
        self.original_matrix_labels = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size):
                label = tk.Label(self.original_section_frame, text="0", 
                               width=5, height=2, relief="solid", borderwidth=1,
                               font=('Courier', 10), bg='white')
                label.grid(row=i, column=j, padx=1, pady=1)
                row.append(label)
            self.original_matrix_labels.append(row)
            
        # Create gradient matrices (Gx, Gy, Magnitude)
        gradient_container = ttk.Frame(self.gradient_section_frame)
        gradient_container.grid(row=0, column=0)
        
        # Gx matrix
        gx_frame = ttk.Frame(gradient_container)
        gx_frame.grid(row=1, column=0, padx=(0, 10))
        ttk.Label(gx_frame, text="Gx (Horizontal)", font=('Arial', 8, 'bold')).grid(row=0, column=0, pady=(0, 2))
        
        self.gx_matrix_labels = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size):
                label = tk.Label(gx_frame, text="0", width=5, height=2, 
                               relief="solid", borderwidth=1, font=('Courier', 9), bg='white')
                label.grid(row=i+1, column=j, padx=1, pady=1)
                row.append(label)
            self.gx_matrix_labels.append(row)
            
        # Gy matrix
        gy_frame = ttk.Frame(gradient_container)
        gy_frame.grid(row=1, column=1, padx=(0, 10))
        ttk.Label(gy_frame, text="Gy (Vertical)", font=('Arial', 8, 'bold')).grid(row=0, column=0, pady=(0, 2))
        
        self.gy_matrix_labels = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size):
                label = tk.Label(gy_frame, text="0", width=5, height=2,
                               relief="solid", borderwidth=1, font=('Courier', 9), bg='white')
                label.grid(row=i+1, column=j, padx=1, pady=1)
                row.append(label)
            self.gy_matrix_labels.append(row)
            
        # Magnitude matrix
        mag_frame = ttk.Frame(gradient_container)
        mag_frame.grid(row=1, column=2)
        ttk.Label(mag_frame, text="Magnitude", font=('Arial', 8, 'bold')).grid(row=0, column=0, pady=(0, 2))
        
        self.mag_matrix_labels = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size):
                label = tk.Label(mag_frame, text="0", width=5, height=2,
                               relief="solid", borderwidth=1, font=('Courier', 9), bg='white')
                label.grid(row=i+1, column=j, padx=1, pady=1)
                row.append(label)
            self.mag_matrix_labels.append(row)
            
        # Create canny result matrix
        self.canny_matrix_labels = []
        for i in range(self.matrix_size):
            row = []
            for j in range(self.matrix_size):
                label = tk.Label(self.canny_section_frame, text="0", 
                               width=5, height=2, relief="solid", borderwidth=1,
                               font=('Courier', 10), bg='white')
                label.grid(row=i, column=j, padx=1, pady=1)
                row.append(label)
            self.canny_matrix_labels.append(row)
            
    def toggle_original_image_section(self):
        self.original_image_expanded.set(not self.original_image_expanded.get())
        if self.original_image_expanded.get():
            self.original_image_frame.grid()
            self.original_image_button.config(text="▼ Original Image (Grayscale)")
        else:
            self.original_image_frame.grid_remove()
            self.original_image_button.config(text="▶ Original Image (Grayscale)")
            
    def toggle_processed_image_section(self):
        self.processed_image_expanded.set(not self.processed_image_expanded.get())
        if self.processed_image_expanded.get():
            self.processed_image_frame.grid()
            self.processed_image_button.config(text="▼ Processed Image (with/without Blur)")
        else:
            self.processed_image_frame.grid_remove()
            self.processed_image_button.config(text="▶ Processed Image (with/without Blur)")
            
    def toggle_canny_image_section(self):
        self.canny_image_expanded.set(not self.canny_image_expanded.get())
        if self.canny_image_expanded.get():
            self.canny_image_frame.grid()
            self.canny_image_button.config(text="▼ Canny Edge Detection")
        else:
            self.canny_image_frame.grid_remove()
            self.canny_image_button.config(text="▶ Canny Edge Detection")
            
    def toggle_original_section(self):
        self.original_expanded.set(not self.original_expanded.get())
        if self.original_expanded.get():
            self.original_section_frame.grid()
            self.original_section_button.config(text="▼ Original Grayscale Values")
        else:
            self.original_section_frame.grid_remove()
            self.original_section_button.config(text="▶ Original Grayscale Values")
            
    def toggle_gradient_section(self):
        self.gradient_expanded.set(not self.gradient_expanded.get())
        if self.gradient_expanded.get():
            self.gradient_section_frame.grid()
            self.gradient_section_button.config(text="▼ Gradient Calculations")
        else:
            self.gradient_section_frame.grid_remove()
            self.gradient_section_button.config(text="▶ Gradient Calculations")
            
    def toggle_canny_section(self):
        self.canny_expanded.set(not self.canny_expanded.get())
        if self.canny_expanded.get():
            self.canny_section_frame.grid()
            self.canny_section_button.config(text="▼ Final Edge Detection")
        else:
            self.canny_section_frame.grid_remove()
            self.canny_section_button.config(text="▶ Final Edge Detection")
        
    def on_kernel_change(self, value):
        # Ensure kernel size is odd
        kernel_size = int(float(value))
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.blur_kernel.set(kernel_size)
        
        self.kernel_value_label.config(text=str(kernel_size))
        self.update_display()
        
    def on_parameter_change(self, value=None):
        # Update value labels
        self.low_value_label.config(text=str(int(self.low_threshold.get())))
        self.high_value_label.config(text=str(int(self.high_threshold.get())))
        
        # Update kernel slider state
        self.update_kernel_state()
        
        # Update display
        self.update_display()
        
    def on_window_change(self, value=None):
        # Update position labels
        self.x_value_label.config(text=str(int(self.window_x.get())))
        self.y_value_label.config(text=str(int(self.window_y.get())))
        
        # Update display
        self.update_display()
        
    def update_kernel_state(self):
        if self.blur_enabled.get():
            self.kernel_scale.config(state='normal')
            self.kernel_value_label.config(foreground='black')
        else:
            self.kernel_scale.config(state='disabled')
            self.kernel_value_label.config(foreground='gray')
            
    def draw_rectangle_on_image(self, image_array, x, y, size):
        """Draw a red rectangle on the image to show the analysis window"""
        # Convert grayscale to RGB for colored rectangle
        if len(image_array.shape) == 2:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image_array.copy()
            
        # Draw rectangle
        cv2.rectangle(rgb_image, (x, y), (x + size, y + size), (255, 0, 0), 2)
        return rgb_image
    
    def compute_gradients(self, image):
        """Compute Sobel gradients"""
        # Sobel X (horizontal edges)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # Sobel Y (vertical edges)  
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # Magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        return gx, gy, magnitude
            
    def update_display(self):
        # Get current window position
        x = int(self.window_x.get())
        y = int(self.window_y.get())
        
        # Display original grayscale image with rectangle
        original_with_rect = self.draw_rectangle_on_image(self.gray_resized, x, y, self.matrix_size)
        original_pil = Image.fromarray(original_with_rect)
        original_photo = ImageTk.PhotoImage(original_pil)
        self.original_label.config(image=original_photo)
        self.original_label.image = original_photo  # Keep a reference
        
        # Apply Gaussian blur if enabled
        processed_image = self.gray_resized.copy()
        if self.blur_enabled.get():
            kernel_size = int(self.blur_kernel.get())
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed_image = cv2.GaussianBlur(processed_image, (kernel_size, kernel_size), 0)
        
        # Display processed image with rectangle
        processed_with_rect = self.draw_rectangle_on_image(processed_image, x, y, self.matrix_size)
        processed_pil = Image.fromarray(processed_with_rect)
        processed_photo = ImageTk.PhotoImage(processed_pil)
        self.processed_label.config(image=processed_photo)
        self.processed_label.image = processed_photo  # Keep a reference
        
        # Compute gradients
        gx, gy, magnitude = self.compute_gradients(processed_image)
        
        # Apply Canny edge detection
        low_thresh = int(self.low_threshold.get())
        high_thresh = int(self.high_threshold.get())
        
        # Ensure high threshold is greater than low threshold
        if high_thresh <= low_thresh:
            high_thresh = low_thresh + 1
            
        canny_edges = cv2.Canny(processed_image, low_thresh, high_thresh)
        
        # Display Canny result with rectangle
        canny_with_rect = self.draw_rectangle_on_image(canny_edges, x, y, self.matrix_size)
        canny_pil = Image.fromarray(canny_with_rect)
        canny_photo = ImageTk.PhotoImage(canny_pil)
        self.canny_label.config(image=canny_photo)
        self.canny_label.image = canny_photo  # Keep a reference
        
        # Update matrices
        self.update_matrices(x, y, processed_image, gx, gy, magnitude, canny_edges)
        
    def update_matrices(self, x, y, processed_image, gx, gy, magnitude, canny_edges):
        """Update all pixel value matrices"""
        # Extract the window regions
        original_window = self.gray_resized[y:y+self.matrix_size, x:x+self.matrix_size]
        gx_window = gx[y:y+self.matrix_size, x:x+self.matrix_size]
        gy_window = gy[y:y+self.matrix_size, x:x+self.matrix_size]
        mag_window = magnitude[y:y+self.matrix_size, x:x+self.matrix_size]
        canny_window = canny_edges[y:y+self.matrix_size, x:x+self.matrix_size]
        
        # Update original values matrix
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if i < original_window.shape[0] and j < original_window.shape[1]:
                    value = int(original_window[i, j])
                    self.original_matrix_labels[i][j].config(text=str(value))
                    # Color code: darker background for higher values
                    intensity = min(255, max(0, value))
                    bg_color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
                    text_color = 'white' if intensity < 128 else 'black'
                    self.original_matrix_labels[i][j].config(bg=bg_color, fg=text_color)
                else:
                    self.original_matrix_labels[i][j].config(text="", bg='white')
                    
        # Update gradient matrices
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if i < gx_window.shape[0] and j < gx_window.shape[1]:
                    # Gx values
                    gx_val = int(gx_window[i, j])
                    self.gx_matrix_labels[i][j].config(text=str(gx_val))
                    # Color code: blue for negative, red for positive, white for zero
                    if gx_val > 0:
                        intensity = min(255, abs(gx_val) * 2)
                        bg_color = f'#{255:02x}{255-intensity//2:02x}{255-intensity//2:02x}'  # Red scale
                        text_color = 'black' if intensity < 128 else 'white'
                    elif gx_val < 0:
                        intensity = min(255, abs(gx_val) * 2)
                        bg_color = f'#{255-intensity//2:02x}{255-intensity//2:02x}{255:02x}'  # Blue scale
                        text_color = 'black' if intensity < 128 else 'white'
                    else:
                        bg_color = 'white'
                        text_color = 'black'
                    self.gx_matrix_labels[i][j].config(bg=bg_color, fg=text_color)
                    
                    # Gy values
                    gy_val = int(gy_window[i, j])
                    self.gy_matrix_labels[i][j].config(text=str(gy_val))
                    # Same color coding as Gx
                    if gy_val > 0:
                        intensity = min(255, abs(gy_val) * 2)
                        bg_color = f'#{255:02x}{255-intensity//2:02x}{255-intensity//2:02x}'  # Red scale
                        text_color = 'black' if intensity < 128 else 'white'
                    elif gy_val < 0:
                        intensity = min(255, abs(gy_val) * 2)
                        bg_color = f'#{255-intensity//2:02x}{255-intensity//2:02x}{255:02x}'  # Blue scale
                        text_color = 'black' if intensity < 128 else 'white'
                    else:
                        bg_color = 'white'
                        text_color = 'black'
                    self.gy_matrix_labels[i][j].config(bg=bg_color, fg=text_color)
                    
                    # Magnitude values
                    mag_val = int(mag_window[i, j])
                    self.mag_matrix_labels[i][j].config(text=str(mag_val))
                    # Grayscale based on magnitude
                    intensity = min(255, max(0, mag_val))
                    bg_color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
                    text_color = 'white' if intensity < 128 else 'black'
                    self.mag_matrix_labels[i][j].config(bg=bg_color, fg=text_color)
                else:
                    self.gx_matrix_labels[i][j].config(text="", bg='white')
                    self.gy_matrix_labels[i][j].config(text="", bg='white')
                    self.mag_matrix_labels[i][j].config(text="", bg='white')
                    
        # Update canny result matrix
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if i < canny_window.shape[0] and j < canny_window.shape[1]:
                    value = int(canny_window[i, j])
                    self.canny_matrix_labels[i][j].config(text=str(value))
                    # Color code: white for edges (255), black for non-edges (0)
                    if value > 0:
                        self.canny_matrix_labels[i][j].config(bg='white', fg='black')
                    else:
                        self.canny_matrix_labels[i][j].config(bg='black', fg='white')
                else:
                    self.canny_matrix_labels[i][j].config(text="", bg='white')
        
    def run(self):
        self.root.mainloop()


def main():
    if len(sys.argv) != 2:
        print("Usage: python canny_demo.py <image_path>")
        print("Example: python canny_demo.py sample.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    app = CannyDemo(image_path)
    app.run()


if __name__ == "__main__":
    main()