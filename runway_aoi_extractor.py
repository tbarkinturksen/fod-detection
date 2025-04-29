import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from datetime import datetime
import tkinter.ttk as ttk

#################################################
# CONFIGURATION SETTINGS
#################################################
# Default patch and window settings
PATCH_SIZE = 512  # Size of extracted patches in pixels
TARGET_WIDTH = 3840  # 4K width for processing
TARGET_HEIGHT = 2160  # 4K height for processing


class PatchSelectorApp:
    """
    GUI application for selecting and extracting patches from runway video frames
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Video Patch Extractor (4K Mode)")

        # Variables
        self.video_path = None
        self.selected_patches = set()
        self.patch_size = PATCH_SIZE
        self.first_frame = None
        self.patch_overlay = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.scale_factor = 1.0
        self.img_width = 0
        self.img_height = 0
        self.target_width = TARGET_WIDTH
        self.target_height = TARGET_HEIGHT

        # Create UI elements
        self.create_widgets()

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

    def create_widgets(self):
        """Create all UI widgets for the application"""
        # Top frame for buttons
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Load video button
        self.load_btn = tk.Button(self.top_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Process video button
        self.process_btn = tk.Button(self.top_frame, text="Process Video", command=self.process_video,
                                     state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Select all button
        self.select_all_btn = tk.Button(self.top_frame, text="Select All Patches", command=self.select_all_patches,
                                        state=tk.DISABLED)
        self.select_all_btn.pack(side=tk.LEFT, padx=5)

        # Clear selection button
        self.clear_btn = tk.Button(self.top_frame, text="Clear Selection", command=self.clear_selection,
                                   state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Format selector
        self.format_label = tk.Label(self.top_frame, text="Output Format:")
        self.format_label.pack(side=tk.LEFT, padx=(20, 5))

        self.format_var = tk.StringVar(value="png")
        self.format_combo = ttk.Combobox(self.top_frame, textvariable=self.format_var,
                                         values=["png", "jpg"], width=5, state="readonly")
        self.format_combo.pack(side=tk.LEFT)

        # Frame skip option
        self.skip_label = tk.Label(self.top_frame, text="Process every")
        self.skip_label.pack(side=tk.LEFT, padx=(20, 5))

        self.skip_var = tk.IntVar(value=1)
        self.skip_spin = tk.Spinbox(self.top_frame, from_=1, to=30, width=2, textvariable=self.skip_var)
        self.skip_spin.pack(side=tk.LEFT)

        self.skip_suffix = tk.Label(self.top_frame, text="frame(s)")
        self.skip_suffix.pack(side=tk.LEFT, padx=5)

        # Add a checkbox for 4K upscaling (always enabled by default)
        self.upscale_var = tk.BooleanVar(value=True)
        self.upscale_check = tk.Checkbutton(self.top_frame, text="Process as 4K", variable=self.upscale_var)
        self.upscale_check.pack(side=tk.LEFT, padx=(20, 5))

        # Canvas for displaying the first frame
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbars
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.v_scrollbar = tk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.canvas_frame, bg="black",
                                xscrollcommand=self.h_scrollbar.set,
                                yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Load a video to begin")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_video(self):
        """Load and process the first frame of the selected video"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not self.video_path:
            return

        # Load the video and extract first frame
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()

            # Get video properties
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            if not ret:
                messagebox.showerror("Error", "Failed to read the first frame from the video")
                return

            # Process as 4K (upscale if needed)
            if self.upscale_var.get() and (
                    self.orig_width < self.target_width or self.orig_height < self.target_height):
                self.first_frame = cv2.resize(frame, (self.target_width, self.target_height),
                                              interpolation=cv2.INTER_CUBIC)
                self.status_var.set(
                    f"Upscaled from {self.orig_width}x{self.orig_height} to {self.target_width}x{self.target_height}")
            else:
                self.first_frame = frame

            # Calculate grid dimensions
            self.img_height, self.img_width = self.first_frame.shape[:2]
            self.grid_rows = self.img_height // self.patch_size
            self.grid_cols = self.img_width // self.patch_size

            # Check if there are any incomplete patches at the edges
            if self.img_height % self.patch_size > 0:
                self.grid_rows += 1
            if self.img_width % self.patch_size > 0:
                self.grid_cols += 1

            self.status_var.set(f"Video loaded: {os.path.basename(self.video_path)} - "
                                f"Processing as {self.img_width}x{self.img_height}, "
                                f"Grid: {self.grid_cols}x{self.grid_rows}, "
                                f"Total patches: {self.grid_rows * self.grid_cols}")

            # Create patch overlay
            self.create_patch_overlay()

            # Enable buttons
            self.process_btn.config(state=tk.NORMAL)
            self.select_all_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")

    def create_patch_overlay(self):
        """Create a grid overlay on the first frame to show patches"""
        if self.first_frame is None:
            return

        # Convert first frame to RGB (from BGR)
        frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)

        # Create a copy for overlay
        self.patch_overlay = frame_rgb.copy()

        # Draw grid lines
        for i in range(1, self.grid_rows):
            y = i * self.patch_size
            if y < self.img_height:
                cv2.line(self.patch_overlay, (0, y), (self.img_width, y), (255, 255, 0), 1)

        for j in range(1, self.grid_cols):
            x = j * self.patch_size
            if x < self.img_width:
                cv2.line(self.patch_overlay, (x, 0), (x, self.img_height), (255, 255, 0), 1)

        # Display the overlay
        self.display_image()

    def display_image(self):
        """Display the image with overlays on canvas"""
        if self.patch_overlay is None:
            return

        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(self.patch_overlay)

        # Calculate scale factor to fit in the canvas frame
        canvas_width = self.canvas_frame.winfo_width() - self.v_scrollbar.winfo_width()
        canvas_height = self.canvas_frame.winfo_height() - self.h_scrollbar.winfo_height()

        if canvas_width > 10 and canvas_height > 10:
            scale_x = canvas_width / self.img_width
            scale_y = canvas_height / self.img_height
            self.scale_factor = min(0.5, min(scale_x, scale_y))  # Limit to 50% max for 4K

            # Resize only if necessary
            if self.scale_factor < 1.0:
                new_width = int(self.img_width * self.scale_factor)
                new_height = int(self.img_height * self.scale_factor)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage
        self.tk_img = ImageTk.PhotoImage(image=pil_img)

        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_window_resize(self, event=None):
        """Handle window resize events"""
        # Only handle resizing if we have an image loaded
        if hasattr(self, 'patch_overlay') and self.patch_overlay is not None:
            # Wait a bit before resizing to avoid too many updates
            self.root.after(100, self.display_image)

    def on_canvas_click(self, event):
        """Handle canvas click events to select patches"""
        if self.patch_overlay is None:
            return

        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.scale_factor < 1.0:
            img_x = int(canvas_x / self.scale_factor)
            img_y = int(canvas_y / self.scale_factor)
        else:
            img_x = int(canvas_x)
            img_y = int(canvas_y)

        # Determine which patch was clicked
        patch_row = img_y // self.patch_size
        patch_col = img_x // self.patch_size

        if 0 <= patch_row < self.grid_rows and 0 <= patch_col < self.grid_cols:
            patch_id = (patch_row, patch_col)

            # Check if this patch is complete (512x512)
            y1 = patch_row * self.patch_size
            x1 = patch_col * self.patch_size
            y2 = min(y1 + self.patch_size, self.img_height)
            x2 = min(x1 + self.patch_size, self.img_width)

            is_complete = (y2 - y1 == self.patch_size and x2 - x1 == self.patch_size)

            if is_complete:
                # Toggle selection
                if patch_id in self.selected_patches:
                    self.selected_patches.remove(patch_id)
                else:
                    self.selected_patches.add(patch_id)

                # Update overlay
                self.update_patch_selection()
            else:
                self.status_var.set(f"Skipping incomplete patch at ({patch_row}, {patch_col})")

    def select_all_patches(self):
        """Select all complete patches in the grid"""
        if self.first_frame is None:
            return

        # Clear existing selection
        self.selected_patches.clear()

        # Add all complete patches
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                y1 = row * self.patch_size
                x1 = col * self.patch_size
                y2 = min(y1 + self.patch_size, self.img_height)
                x2 = min(x1 + self.patch_size, self.img_width)

                if y2 - y1 == self.patch_size and x2 - x1 == self.patch_size:
                    self.selected_patches.add((row, col))

        self.update_patch_selection()

    def clear_selection(self):
        """Clear all selected patches"""
        self.selected_patches.clear()
        self.update_patch_selection()

    def update_patch_selection(self):
        """Update the visual overlay to show selected patches"""
        if self.first_frame is None:
            return

        # Start with fresh overlay
        self.patch_overlay = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB).copy()

        # Draw grid lines
        for i in range(1, self.grid_rows):
            y = i * self.patch_size
            if y < self.img_height:
                cv2.line(self.patch_overlay, (0, y), (self.img_width, y), (255, 255, 0), 1)

        for j in range(1, self.grid_cols):
            x = j * self.patch_size
            if x < self.img_width:
                cv2.line(self.patch_overlay, (x, 0), (x, self.img_height), (255, 255, 0), 1)

        # Highlight selected patches
        for row, col in self.selected_patches:
            y1 = row * self.patch_size
            x1 = col * self.patch_size
            y2 = min(y1 + self.patch_size, self.img_height)
            x2 = min(x1 + self.patch_size, self.img_width)

            # Create semi-transparent overlay
            overlay = self.patch_overlay[y1:y2, x1:x2].copy()
            cv2.rectangle(self.patch_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Apply green tint to selected patches
            green_overlay = np.zeros_like(overlay)
            green_overlay[:] = (0, 200, 0)
            cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0, self.patch_overlay[y1:y2, x1:x2])

        # Display updated overlay
        self.display_image()
        self.status_var.set(f"Selected patches: {len(self.selected_patches)}")

    def process_video(self):
        """Process the video and extract selected patches from all frames"""
        if not self.selected_patches:
            messagebox.showwarning("Warning", "No patches selected. Please select at least one patch.")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = f"patches_{os.path.basename(self.video_path).split('.')[0]}_{timestamp}"
        output_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Open the video file
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Failed to open the video file")
                return

            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing Video")
            progress_window.geometry("400x150")
            progress_window.resizable(False, False)
            progress_window.grab_set()  # Make it modal

            # Progress widgets
            frame_var = tk.StringVar(value="Processing frame: 0 / 0")
            patch_var = tk.StringVar(value="Extracted patches: 0")

            tk.Label(progress_window, text="Processing video...").pack(pady=(10, 5))
            tk.Label(progress_window, textvariable=frame_var).pack(pady=2)
            tk.Label(progress_window, textvariable=patch_var).pack(pady=2)

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
            progress_bar.pack(pady=10)

            cancel_btn = tk.Button(progress_window, text="Cancel", command=progress_window.destroy)
            cancel_btn.pack(pady=5)

            # Process video frames
            frame_idx = 0
            frame_count = 0
            patch_count = 0
            frame_skip = self.skip_var.get()
            patch_format = self.format_var.get()
            processing_cancelled = False

            # Create log file
            log_path = os.path.join(output_dir, "processing_log.txt")
            with open(log_path, 'w') as log_file:
                log_file.write(f"Video: {self.video_path}\n")
                log_file.write(f"Original dimensions: {self.orig_width}x{self.orig_height}\n")
                log_file.write(f"Processing dimensions: {self.img_width}x{self.img_height}\n")
                log_file.write(f"Patch size: {self.patch_size}x{self.patch_size}\n")
                log_file.write(f"Selected patches: {len(self.selected_patches)}\n")
                log_file.write(f"FPS: {self.fps}\n")
                log_file.write(f"Total frames: {self.total_frames}\n")
                log_file.write(f"Processing every {frame_skip} frame(s)\n\n")
                log_file.write("Processing started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

                # Store selected patch coordinates for reference
                log_file.write("Selected patch coordinates (row, col):\n")
                for row, col in sorted(self.selected_patches):
                    y1 = row * self.patch_size
                    x1 = col * self.patch_size
                    log_file.write(f"({row}, {col}) - Top-left pixel: ({x1}, {y1})\n")
                log_file.write("\n")

                # Process each frame
                while True:
                    # Check if cancelled
                    progress_window.update()
                    if not progress_window.winfo_exists():
                        processing_cancelled = True
                        break

                    # Read next frame
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process only every n-th frame
                    if frame_idx % frame_skip == 0:
                        # Upscale to 4K if needed
                        if self.upscale_var.get() and (
                                frame.shape[1] != self.target_width or frame.shape[0] != self.target_height):
                            frame = cv2.resize(frame, (self.target_width, self.target_height),
                                               interpolation=cv2.INTER_CUBIC)

                        # Extract selected patches
                        for row, col in self.selected_patches:
                            y1 = row * self.patch_size
                            x1 = col * self.patch_size
                            y2 = min(y1 + self.patch_size, frame.shape[0])
                            x2 = min(x1 + self.patch_size, frame.shape[1])

                            # Skip incomplete patches (should not happen if selection is valid)
                            if y2 - y1 != self.patch_size or x2 - x1 != self.patch_size:
                                continue

                            patch = frame[y1:y2, x1:x2]
                            patch_filename = f"frame_{frame_idx:06d}_patch_{row:02d}_{col:02d}.{patch_format}"

                            if patch_format == "jpg":
                                cv2.imwrite(os.path.join(output_dir, patch_filename), patch,
                                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                            else:
                                cv2.imwrite(os.path.join(output_dir, patch_filename), patch)

                            patch_count += 1

                        # Update progress display
                        frame_count += 1
                        progress_var.set((frame_idx / self.total_frames) * 100)
                        frame_var.set(f"Processing frame: {frame_idx} / {self.total_frames}")
                        patch_var.set(f"Extracted patches: {patch_count}")

                        # Update less frequently for better performance
                        if frame_count % 5 == 0:
                            progress_window.update()

                    # Increment frame index
                    frame_idx += 1

                # Finalize log
                log_file.write(f"Processing ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Processed frames: {frame_count} (from {frame_idx} total frames)\n")
                log_file.write(f"Extracted patches: {patch_count}\n")
                if processing_cancelled:
                    log_file.write("Processing was cancelled by user\n")

            # Clean up
            cap.release()

            # Close progress window if still open
            if progress_window.winfo_exists():
                progress_window.destroy()

            # Show completion message
            if not processing_cancelled:
                messagebox.showinfo("Processing Complete",
                                    f"Processed {frame_count} frames (from {frame_idx} total frames)\n"
                                    f"Extracted {patch_count} patches\n"
                                    f"Saved to: {output_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    root.title("Video Patch Extractor (4K Mode)")
    root.geometry("1280x800")
    app = PatchSelectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
