"""Professional video viewer for displaying frames without distortion."""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class VideoViewer(tk.Frame):
    """Professional video viewer with letterboxing support."""

    def __init__(self, master, width=640, height=480, bg="#404040", **kwargs):
        """
        Initialize a professional video viewer.
        
        Args:
            master: Parent widget
            width: Display width (pixels) - 640
            height: Display height (pixels) - 480
            bg: Background color
        """
        # Extract bg from kwargs if present (avoid duplicate keyword argument)
        bg = kwargs.pop('bg', bg)
        super().__init__(master, bg=bg, **kwargs)
        self.width = width
        self.height = height
        
        # Main label for display
        self.label = tk.Label(self, bg=bg)
        self.label.pack(expand=True, fill="both")
        
        self.current_photo = None
        self._create_placeholder()
    
    def _create_placeholder(self):
        """Create a gray placeholder image."""
        placeholder = Image.new("RGB", (self.width, self.height), color=(64, 64, 64))
        self.current_photo = ImageTk.PhotoImage(placeholder)
        self.label.config(image=self.current_photo)
    
    def display_image(self, cv_image):
        """
        Display a CV2/numpy image (BGR format) in the viewer with letterboxing.
        
        Args:
            cv_image: numpy array in BGR format (OpenCV/cv2 format)
        """
        if cv_image is None:
            self._create_placeholder()
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize with letterboxing (no distortion)
        resized = self._resize_letterbox(rgb_image, self.width, self.height)
        
        # Convert to PhotoImage and display
        pil_image = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(pil_image)
        self.label.config(image=self.current_photo)
    
    @staticmethod
    def _resize_letterbox(cv_image, target_w, target_h):
        """
        Resize image with letterboxing to maintain aspect ratio.
        
        Args:
            cv_image: Input BGR image (numpy array)
            target_w: Target width
            target_h: Target height
            
        Returns:
            Resized image with letterboxing (numpy array in RGB)
        """
        h, w = cv_image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create letterbox canvas (dark gray)
        canvas = 64 * np.ones((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

