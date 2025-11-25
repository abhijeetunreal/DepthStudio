import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
import requests
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- CONFIGURATION ---
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
MODEL_FILENAME = "depth_anything_v2_small_f32.onnx"
INPUT_SIZE = 518 

class UnifiedDepthApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Depth Studio - GPU/CPU Switcher")
        self.root.geometry("1050x700")
        self.root.configure(bg="#1e272e")

        # --- Variables ---
        self.calibration_mode = "MANUAL" 
        self.ref_roi = None              
        self.ref_real_dist = 2.0         
        self.is_selecting = False        
        self.select_start = (0,0)
        self.select_end = (0,0)
        self.show_pip = tk.BooleanVar(value=True)
        
        # Camera Variables
        self.cap = None
        self.cam_exposure = tk.IntVar(value=-5)
        self.cam_gain = tk.IntVar(value=0)

        # AI Session Placeholder
        self.session = None
        self.input_name = ""
        
        # --- UI LAYOUT ---
        # Left: Video Feed
        self.video_frame = tk.Frame(self.root, bg="black", width=640, height=480)
        self.video_frame.pack(side="left", padx=10, pady=10)
        
        self.lbl_video = tk.Label(self.video_frame, bg="black")
        self.lbl_video.pack()
        
        self.lbl_video.bind("<Button-1>", self.on_mouse_down)
        self.lbl_video.bind("<B1-Motion>", self.on_mouse_drag)
        self.lbl_video.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Right: Controls
        self.control_panel = tk.Frame(self.root, bg="#1e272e", width=300)
        self.control_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.build_controls()
        
        # Initialize AI (Try GPU first by default)
        self.switch_processor("GPU")
        self.init_mediapipe()

        # Start System
        self.start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.video_loop()
        self.root.mainloop()

    def init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

    def build_controls(self):
        LBL_STYLE = {"bg": "#1e272e", "fg": "white", "font": ("Segoe UI", 10)}
        HEADER_STYLE = {"bg": "#1e272e", "fg": "#00d2d3", "font": ("Segoe UI", 12, "bold")}
        
        # --- SECTION 0: PROCESSOR SWITCHER (NEW) ---
        tk.Label(self.control_panel, text="HARDWARE ACCELERATION", **HEADER_STYLE).pack(pady=(0, 10))
        
        proc_frame = tk.LabelFrame(self.control_panel, text="Select Processor", bg="#1e272e", fg="white")
        proc_frame.pack(fill="x", pady=5)

        self.lbl_processor = tk.Label(proc_frame, text="Current: INITIALIZING", font=("Segoe UI", 10, "bold"), bg="#1e272e", fg="#f1c40f")
        self.lbl_processor.pack(pady=5)

        btn_row = tk.Frame(proc_frame, bg="#1e272e")
        btn_row.pack(pady=5)

        tk.Button(btn_row, text="SWITCH TO GPU", bg="#2ecc71", fg="white", font=("Segoe UI", 9, "bold"), 
                 command=lambda: self.switch_processor("GPU")).pack(side="left", padx=5)
        
        tk.Button(btn_row, text="SWITCH TO CPU", bg="#3498db", fg="white", font=("Segoe UI", 9, "bold"), 
                 command=lambda: self.switch_processor("CPU")).pack(side="left", padx=5)

        # --- SECTION 1: CAMERA EXPOSURE ---
        tk.Label(self.control_panel, text="CAMERA SETTINGS", **HEADER_STYLE).pack(pady=(20, 10))
        cam_frame = tk.LabelFrame(self.control_panel, text="Lock Exposure", bg="#1e272e", fg="white")
        cam_frame.pack(fill="x", pady=5)

        tk.Label(cam_frame, text="Exposure (Left = Darker/Stable)", **LBL_STYLE).pack(anchor="w")
        self.scale_exp = tk.Scale(cam_frame, from_=-13, to=0, orient="horizontal", 
                                  variable=self.cam_exposure, bg="#1e272e", fg="white", 
                                  command=self.update_camera_settings)
        self.scale_exp.pack(fill="x", padx=5)

        tk.Label(cam_frame, text="Gain (ISO)", **LBL_STYLE).pack(anchor="w")
        self.scale_gain = tk.Scale(cam_frame, from_=0, to=255, orient="horizontal", 
                                   variable=self.cam_gain, bg="#1e272e", fg="white",
                                   command=self.update_camera_settings)
        self.scale_gain.pack(fill="x", padx=5)

        # --- SECTION 2: CALIBRATION ---
        tk.Label(self.control_panel, text="CALIBRATION", **HEADER_STYLE).pack(pady=(20, 10))
        cal_frame = tk.LabelFrame(self.control_panel, text="Reference Object", bg="#1e272e", fg="white")
        cal_frame.pack(fill="x", pady=5)

        input_row = tk.Frame(cal_frame, bg="#1e272e")
        input_row.pack(fill="x", pady=5)
        
        tk.Label(input_row, text="Dist (m):", **LBL_STYLE).pack(side="left")
        self.entry_dist = tk.Entry(input_row, width=6, justify="center", font=("Segoe UI", 12))
        self.entry_dist.insert(0, "2.0")
        self.entry_dist.pack(side="left", padx=5)
        tk.Button(input_row, text="Update", bg="#f39c12", fg="white", command=self.update_distance).pack(side="left")

        self.lbl_mode = tk.Label(cal_frame, text="Mode: MANUAL", font=("Segoe UI", 10, "bold"), bg="#1e272e", fg="#ff6b6b")
        self.lbl_mode.pack(pady=10)

        tk.Button(cal_frame, text="[ DRAW BOX MODE ]", bg="#2e86de", fg="white", font=("Segoe UI", 10, "bold"), command=self.enable_drawing).pack(fill="x", padx=5, pady=5)
        tk.Button(cal_frame, text="Clear Reference", bg="#ee5253", fg="white", command=self.clear_roi).pack(fill="x", padx=5, pady=5)

        # --- SECTION 3: DATA ---
        tk.Label(self.control_panel, text="LIVE DATA", **HEADER_STYLE).pack(pady=(20, 10))
        self.lbl_ref_val = tk.Label(self.control_panel, text="Ref Raw: ---", **LBL_STYLE)
        self.lbl_ref_val.pack()

        tk.Checkbutton(self.control_panel, text="Show Robot Vision (PIP)", variable=self.show_pip, bg="#1e272e", fg="white", selectcolor="#1e272e").pack(pady=20)

    def switch_processor(self, target_mode):
        """Switches between CPU and GPU safely at runtime."""
        # 1. Check model file
        if not os.path.exists(MODEL_FILENAME):
            self.lbl_processor.config(text="DOWNLOADING MODEL...", fg="#f1c40f")
            self.root.update()
            try:
                r = requests.get(MODEL_URL)
                with open(MODEL_FILENAME, 'wb') as f: f.write(r.content)
            except Exception as e:
                messagebox.showerror("Error", f"Download failed: {e}")
                return

        # 2. Attempt Switch
        providers = []
        if target_mode == "GPU":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Fallback to CPU if CUDA not available
        else:
            providers = ['CPUExecutionProvider']

        try:
            # Force restart of session
            old_session = self.session
            self.session = None # Pause inference in loop
            
            new_session = ort.InferenceSession(MODEL_FILENAME, providers=providers)
            self.input_name = new_session.get_inputs()[0].name
            self.session = new_session # Resume inference
            
            # Verify what actually loaded
            active = self.session.get_providers()[0]
            available = self.session.get_providers()
            
            print(f"Available providers: {available}")
            print(f"Active provider: {active}")
            
            if target_mode == "GPU" and 'CUDA' not in active:
                raise Exception(f"GPU requested but loaded: {active}")

            self.lbl_processor.config(text=f"Current: {target_mode} (Active)", fg="#2ecc71")
            print(f"✓ Successfully switched to {target_mode}")

        except Exception as e:
            print(f"✗ Switch failed: {e}")
            if target_mode == "GPU":
                messagebox.showwarning("GPU Failed", f"Could not load CUDA/GPU.\nError: {e}\n\nFalling back to CPU.\n\nTo fix:\n1. Install NVIDIA Drivers\n2. Run: pip install onnxruntime-gpu")
                self.switch_processor("CPU") # Fallback
            else:
                messagebox.showerror("Fatal Error", "Could not load CPU provider.")

    def start_camera(self):
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640); self.cap.set(4, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.update_camera_settings(None)

    def update_camera_settings(self, event):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.cam_exposure.get())
            self.cap.set(cv2.CAP_PROP_GAIN, self.cam_gain.get())

    def on_mouse_down(self, event):
        if self.calibration_mode == "SELECTING":
            self.is_selecting = True; self.select_start = (event.x, event.y); self.select_end = (event.x, event.y)

    def on_mouse_drag(self, event):
        if self.is_selecting: self.select_end = (event.x, event.y)

    def on_mouse_up(self, event):
        if self.is_selecting:
            self.is_selecting = False; self.select_end = (event.x, event.y); self.finalize_roi()

    def enable_drawing(self):
        self.update_distance()
        self.calibration_mode = "SELECTING"
        self.lbl_mode.config(text="ACTION: Draw Box >", fg="#f1c40f")

    def update_distance(self):
        try: self.ref_real_dist = float(self.entry_dist.get())
        except: pass

    def finalize_roi(self):
        x1, y1 = self.select_start; x2, y2 = self.select_end
        x, w = min(x1, x2), abs(x1 - x2); y, h = min(y1, y2), abs(y1 - y2)
        if w > 10 and h > 10:
            self.ref_roi = (x, y, w, h)
            self.calibration_mode = "REFERENCE"
            self.lbl_mode.config(text="Mode: ACTIVE (Locked)", fg="#1dd1a1")
        else:
            self.lbl_mode.config(text="Box too small", fg="#ff6b6b")

    def clear_roi(self):
        self.ref_roi = None; self.calibration_mode = "MANUAL"; self.lbl_mode.config(text="Mode: MANUAL", fg="#ff6b6b")

    def preprocess(self, frame):
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        image = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)

    def video_loop(self):
        ret, frame = self.cap.read()
        if not ret: self.on_close(); return
        display_frame = frame.copy(); h, w = frame.shape[:2]

        # 1. Inference (Only if session is loaded)
        if self.session:
            try:
                depth_raw = self.session.run(None, {self.input_name: self.preprocess(frame)})[0]
                depth_map = cv2.resize(np.squeeze(depth_raw), (w, h))

                # 2. Logic
                current_k = 50.0
                if self.calibration_mode == "REFERENCE" and self.ref_roi:
                    rx, ry, rw, rh = self.ref_roi
                    rx=max(0,rx); ry=max(0,ry); rw=min(w-rx,rw); rh=min(h-ry,rh)
                    ref_crop = depth_map[ry:ry+rh, rx:rx+rw]
                    if ref_crop.size > 0:
                        ref_val = np.median(ref_crop)
                        self.lbl_ref_val.config(text=f"Ref Raw: {ref_val:.2f}")
                        current_k = ref_val * self.ref_real_dist
                        cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (255, 150, 0), 2)
                        cv2.putText(display_frame, f"REF: {self.ref_real_dist}m", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,150,0), 1)

                # 3. Head Tracking
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose = self.pose.process(rgb)
                if pose.pose_landmarks:
                    lm = pose.pose_landmarks.landmark[0]
                    nx, ny = int(lm.x * w), int(lm.y * h)
                    if 0 <= nx < w and 0 <= ny < h:
                        user_raw = depth_map[ny, nx] + 0.01
                        user_dist = current_k / user_raw
                        color = (0, 255, 0) if self.calibration_mode == "REFERENCE" else (0, 100, 255)
                        cv2.circle(display_frame, (nx, ny), 8, color, -1)
                        cv2.putText(display_frame, f"{user_dist:.2f} m", (nx+15, ny), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                if self.show_pip.get():
                    d_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
                    d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                    sw, sh = int(w*0.25), int(h*0.25)
                    display_frame[h-sh:h, w-sw:w] = cv2.resize(d_color, (sw, sh))
                    cv2.rectangle(display_frame, (w-sw, h-sh), (w, h), (200,200,200), 1)

            except Exception as e:
                print(f"Inference Error (Switching?): {e}")

        if self.is_selecting:
            cv2.rectangle(display_frame, self.select_start, self.select_end, (0, 255, 255), 2)

        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.lbl_video.imgtk = imgtk
        self.lbl_video.configure(image=imgtk)
        self.root.after(10, self.video_loop)

    def on_close(self):
        if self.cap: self.cap.release()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    UnifiedDepthApp()