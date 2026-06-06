import threading
import time
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv 

# Importando as classes corrigidas diretamente do seu arquivo world.py
from src.models.world import World, Entidy, Robot
from src.gui.viewers import VideoViewer
from src.detec.detector import detect_centroid


import traceback

# Colors (B,G,R)
RED_COLOR     = (0,0,255) 
BLUE_COLOR    = (255,0,0)
GREEN_COLOR   = (0,255,0)
YELLOW_COLOR  = (255,255,0)

class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Kalman - Demo")
        
        # Video display size
        self.video_width = 640
        self.video_height = 480

        # Set minimum window size
        self.root.minsize(1400, 750)
        
        # Dimensões reais desejadas
        self.max_x = 100 #metrosf
        self.max_y = 75 #metros

        # Localização das torres no mapa
        self.towers = None 

        #Tamanho mínimo do ROI 
        self.min_window_m = int(self.max_x/60) #metros

        # Maximize window
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-fullscreen", True)
            except Exception:
                pass
        self.root.bind("<Escape>", self._on_escape)

        # Application state
        self.video_path = None
        self.processed_video_path = None
        self.cap = None
        self.worker = None
        self.running = False
        self.processing = False
        self.playing = False
        self.paused = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.detection_rate = 0.0  # Métrica de % de detecção estável
        self.meas_inside_roi = 0 

        # Metrics data (Agora armazenados em METROS para os gráficos)
        self.meas_pts = []
        self.filt_pts = []
        self.sqerr_x = []
        self.sqerr_y = []
        self.kalman_windows = []  # Armazena as matrizes P de covariância
        self.pred_pts = []          # posição predita (antes do update)
        self.innovations = []       # diferença z - H*x_pred (2D)
        self.nis_vals = []          # Corrigido para uniformidade com o processamento
        self.prior_covs = []        # covariância predita P_pred
        self.measurements_raw = []  # medições em metros (para gráficos de dispersão)

        self.load_interface()

    def load_interface(self):
        # Visualization toggle states
        self.show_traj = tk.BooleanVar(value=True)
        self.show_detect = tk.BooleanVar(value=True)
        self.show_kalman = tk.BooleanVar(value=True)
        self.show_window = tk.BooleanVar(value=True)

        # Thread-safe boolean cache para o processador de vídeo
        self.show_traj_val = True
        self.show_detect_val = True
        self.show_kalman_val = True
        self.show_window_val = True

        # Main layout: left (300) | center (750) | right (450) = proportion 2:5:3
        self.root.grid_columnconfigure(0, weight=0, minsize=300)  
        self.root.grid_columnconfigure(1, weight=1, minsize=750)  
        self.root.grid_columnconfigure(2, weight=0, minsize=450)  
        self.root.grid_rowconfigure(0, weight=1)

        # build painel
        self._build_left_painel()
        self._build_center_painer()
        self._build_right_painel()

        # Playback poll
        self.root.after(33, self._poll_playback)

    def _build_left_painel(self):
        ### --- LEFT PANEL: 7 Sections (Com Métricas e Torres) ---
        self.root.grid_rowconfigure(0, weight=1) 
        
        self.left_frame = tk.Frame(self.root, bg="white", width=300)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)

        # Título Super Compacto
        title_lbl = tk.Label(self.left_frame, text="Filtro de Kalman", 
                             font=("Segoe UI", 11, "bold"), bg="white", fg="#333333")
        title_lbl.pack(fill="x", padx=10, pady=(4, 2))
        
        # Separador Fino
        sep1 = tk.Frame(self.left_frame, bg="#e0e0e0", height=1)
        sep1.pack(fill="x", padx=10, pady=2)

        # ===== SECTION 1: Carregar o vídeo =====
        entrada_lbl_frame = tk.LabelFrame(self.left_frame, text="📄 Carregar o vídeo", 
                                         font=("Segoe UI", 9, "bold"), 
                                         bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        entrada_lbl_frame.pack(fill="x", padx=8, pady=1)

        self.video_path_label = tk.Label(entrada_lbl_frame, text="Nenhum vídeo (Ou Arena Sintética)", 
                                         wraplength=260, font=("Segoe UI", 8), 
                                         fg="#666666", bg="#f5f5f5")
        self.video_path_label.pack(anchor="w", pady=(1, 2))

        self.load_btn = tk.Button(entrada_lbl_frame, text="📁 Load Vídeo", command=self.load_video, 
                                font=("Segoe UI", 8, "bold"), bg="#4a4a4a", fg="white", 
                                relief="flat", padx=10, pady=1, cursor="hand2")
        self.load_btn.pack(fill="x", pady=1)

        # ===== SECTION 2: Opções do Filtro =====
        config_lbl_frame = tk.LabelFrame(self.left_frame, text="⚙ Opções do Filtro & Sensor", 
                                         font=("Segoe UI", 9, "bold"), 
                                         bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        config_lbl_frame.pack(fill="x", padx=8, pady=1)

        self.detector_noise_entry = ttk.Entry(config_lbl_frame, width=10)
        self.detector_noise_entry.insert(0, "0.16")
        
        lbl_noise = ttk.Label(config_lbl_frame, text="Ruído do Sensor (Metros):", font=("Segoe UI", 8))
        lbl_noise.pack(anchor="w", pady=(1, 0))
        self.detector_noise_entry.pack(anchor="w", pady=(0, 2))

        ttk.Label(config_lbl_frame, text="Q - Diagonal (Processo):", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=1)
        
        q_frame = tk.Frame(config_lbl_frame, bg="#f5f5f5")
        q_frame.pack(fill="x", pady=(0, 1))
        
        q_labels = ["Q[0,0]", "Q[1,1]", "Q[2,2]", "Q[3,3]", "Q[4,4]", "Q[5,5]"]
        default_q_vals = ["5", "5", "5", "5", "5", "5"]
        
        self.q_entries = []
        for i, label in enumerate(q_labels):
            c = i % 3
            r = i // 3
            lbl = tk.Label(q_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=r*2, column=c, sticky="w", padx=1, pady=0)
            
            entry = ttk.Entry(q_frame, width=8)
            entry.insert(0, default_q_vals[i])
            entry.grid(row=r*2+1, column=c, sticky="ew", padx=1, pady=(0, 1))
            self.q_entries.append(entry)
            
        q_frame.columnconfigure(0, weight=1)
        q_frame.columnconfigure(1, weight=1)
        q_frame.columnconfigure(2, weight=1)

        ttk.Label(config_lbl_frame, text="R - Diagonal (Medição):", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=1)
        
        r_frame = tk.Frame(config_lbl_frame, bg="#f5f5f5")
        r_frame.pack(fill="x", pady=(0, 1))
        
        r_labels = ["R[0,0]", "R[1,1]"]
        default_r_vals = ["1e-1", "1e-1"]
        
        self.r_entries = []
        for i, label in enumerate(r_labels):
            c = i % 2 
            r = i // 2
            lbl = tk.Label(r_frame, text=label, font=("Segoe UI", 7), bg="#f5f5f5")
            lbl.grid(row=r*2, column=c, sticky="w", padx=1, pady=0)
            
            entry = ttk.Entry(r_frame, width=8)
            entry.insert(0, default_r_vals[i])
            entry.grid(row=r*2+1, column=c, sticky="ew", padx=1, pady=(0, 1))
            self.r_entries.append(entry)
            
        r_frame.columnconfigure(0, weight=1)
        r_frame.columnconfigure(1, weight=1)

        # ===== SECTION 3: Opções de Debug =====
        debug_lbl_frame = tk.LabelFrame(self.left_frame, text="🔍 Opções de Debug", 
                                        font=("Segoe UI", 9, "bold"), 
                                        bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        debug_lbl_frame.pack(fill="x", padx=8, pady=1)

        ttk.Checkbutton(debug_lbl_frame, text="Desenhar trajetória", variable=self.show_traj).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar detecção", variable=self.show_detect).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Kalman", variable=self.show_kalman).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Janela Kalman", variable=self.show_window).pack(anchor="w")

        # ===== SECTION 4: Status e Métricas =====
        info_lbl_frame = tk.LabelFrame(self.left_frame, text="📊 Status e Métricas", 
                                        font=("Segoe UI", 9, "bold"), 
                                        bg="#f5f5f5", fg="#333333", padx=8, pady=4)
        info_lbl_frame.pack(fill="x", padx=8, pady=2)

        # Status geral com destaque em negrito
        self.status_lbl = tk.Label(info_lbl_frame, text="Status: Aguardando Ação", 
                                font=("Segoe UI", 8, "bold"), fg="#16a34a", bg="#f5f5f5")
        self.status_lbl.pack(anchor="w", pady=(0, 2))

        self.det_rate_lbl = tk.Label(info_lbl_frame, text="Taxa de Detecção: --%", 
                                    font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.det_rate_lbl.pack(anchor="w", pady=1)

        self.inlier_rate_lbl = tk.Label(info_lbl_frame, text="Inliers (na Janela): --%", 
                                        font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.inlier_rate_lbl.pack(anchor="w", pady=1)

        self.rmse_lbl = tk.Label(info_lbl_frame, text="RMSE (X | Y): -- | -- m", 
                                 font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.rmse_lbl.pack(anchor="w", pady=1)

        self.mean_err_lbl = tk.Label(info_lbl_frame, text="Erro Médio (X | Y): -- | -- m", 
                                     font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.mean_err_lbl.pack(anchor="w", pady=1)

        self.nis_lbl = tk.Label(info_lbl_frame, text="NIS Médio: --", 
                               font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.nis_lbl.pack(anchor="w", pady=1)

        # ADICIONADO: Nova label para Regime Estacionário
        self.steady_state_lbl = tk.Label(info_lbl_frame, text="Regime Estacionário: -- s (-- frames)", 
                                        font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.steady_state_lbl.pack(anchor="w", pady=1)

        # ===== SPACER: Garante o uso de todo o height útil =====
        spacer = tk.Frame(self.left_frame, bg="white")
        spacer.pack(fill="both", expand=True)

        # ===== SECTION 5: EXEC & SAVE Buttons =====
        exec_lbl_frame = tk.Frame(self.left_frame, bg="white")
        exec_lbl_frame.pack(side="bottom", fill="x", padx=8, pady=(4, 8))

        btn_frame = tk.Frame(exec_lbl_frame, bg="white")
        btn_frame.pack(fill="x")

        self.exec_btn = tk.Button(btn_frame, text="▶ EXEC", command=self.execute_processing, 
                                font=("Segoe UI", 9, "bold"), bg="#333333", fg="white", 
                                relief="flat", padx=10, pady=8, cursor="hand2", state="disabled")
        self.exec_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.save_btn = tk.Button(btn_frame, text="💾 Salvar", command=self.save_results, 
                                font=("Segoe UI", 9, "bold"), bg="#666666", fg="white", 
                                relief="flat", padx=10, pady=8, cursor="hand2", state="disabled")
        self.save_btn.pack(side="right", fill="x", expand=True, padx=(3, 0))

    def _build_center_painer(self):
        # --- CENTER PANEL: Single Viewer ---
        self.center_frame = tk.Frame(self.root, bg="#f3f4f6")
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # 1. TÍTULO
        viewer_title = tk.Label(self.center_frame, text="🎬 Rastreamento com Filtro de Kalman", 
                            font=("Segoe UI", 14, "bold"), bg="#f3f4f6", fg="#1f2937")
        viewer_title.pack(fill="x", pady=(0, 10))
        
        # 2. VIEWER CONTAINER
        viewer_container = tk.Frame(self.center_frame, bg="#000000", bd=0, highlightthickness=1, highlightbackground="#d1d5db")
        viewer_container.pack(fill="both", expand=True, pady=(0, 15))
        
        self.tela_viewer = VideoViewer(viewer_container, width=self.video_width, 
                                    height=self.video_height, bg="black")
        self.tela_viewer.pack(expand=True)

        # 3. PLAYBACK CONTROLS
        controls_frame = tk.Frame(self.center_frame, bg="#f3f4f6")
        controls_frame.pack(fill="x", pady=(0, 15))
        
        controls_inner = tk.Frame(controls_frame, bg="#f3f4f6")
        controls_inner.pack(anchor="center")
        
        btn_opts = {
            "font": ("Segoe UI", 10, "bold"), "bg": "#374151", "fg": "white", 
            "relief": "flat", "padx": 16, "pady": 6, "cursor": "hand2",
            "activebackground": "#4b5563", "borderwidth": 0
        }
        
        self.prev_btn = tk.Button(controls_inner, text="◄ Anterior", command=self.prev_frame, state="disabled", **btn_opts)
        self.prev_btn.grid(row=0, column=0, padx=8)
        
        self.play_btn = tk.Button(controls_inner, text="⏵ Play", command=self.toggle_playback, state="disabled", **btn_opts)
        self.play_btn.grid(row=0, column=1, padx=8)
        
        self.next_btn = tk.Button(controls_inner, text="Próximo ►", command=self.next_frame, state="disabled", **btn_opts)
        self.next_btn.grid(row=0, column=2, padx=8)

        self.time_info_lbl = tk.Label(controls_inner, text="00:00 / 00:00", 
                                    font=("Segoe UI", 11, "bold"), fg="#374151", bg="#f3f4f6")
        self.time_info_lbl.grid(row=0, column=3, padx=(20, 0))

        # 4. DASHBOARD DE INFORMAÇÕES
        info_dashboard = tk.Frame(self.center_frame, bg="#ffffff", bd=1, relief="solid", highlightthickness=0)
        info_dashboard.config(highlightbackground="#d1d5db", highlightcolor="#d1d5db")
        info_dashboard.pack(fill="x", pady=(0, 0))
        
        # 4.1 Banner Principal do Vídeo
        self.video_info_lbl = tk.Label(info_dashboard, 
                                    text="Arquivo: —  |  Tamanho: —  |  FPS: —  |  Frames: —  |  Taxa: —",
                                    font=("Segoe UI", 9, "bold"), fg="#374151", bg="#ffffff")
        self.video_info_lbl.pack(fill="x", pady=(12, 6))
        
        tk.Frame(info_dashboard, bg="#e5e7eb", height=1).pack(fill="x", padx=30, pady=4)
        
        # 4.2 Container da Legenda
        legend_container = tk.Frame(info_dashboard, bg="#ffffff")
        legend_container.pack(fill="x", pady=(6, 12))
        
        # Escala: quantos pixels representam 1 metro (px/m)
        px_m_x = self.video_width / self.max_x if hasattr(self, 'max_x') and self.max_x > 0 else 1.0
        px_m_y = self.video_height / self.max_y if hasattr(self, 'max_y') and self.max_y > 0 else 1.0
        avg_scale_px_m = (px_m_x + px_m_y) / 2.0
        
        # Obtém o ruído configurado em METROS
        try:
            erro_metros = float(self.detector_noise_entry.get().strip())
        except Exception:
            erro_metros = 1.0
        
        # Converte o erro de metros para pixels
        erro_px = erro_metros * avg_scale_px_m
        
        # Linha 1: Identificação Visual (Cores)
        l1_frame = tk.Frame(legend_container, bg="#ffffff")
        l1_frame.pack(anchor="center", pady=2)
        tk.Label(l1_frame, text="🔴 Detecção: Medido", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#dc2626").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🟦 Kalman: Estimado", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#2563eb").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🟩 Trajetória: Filtrado", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#16a34a").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🔷 Janela: Incerteza (±3σ)", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#9333ea").pack(side="left", padx=15)

        # Linha 2: Métricas de Escala e Debug
        l2_frame = tk.Frame(legend_container, bg="#ffffff")
        l2_frame.pack(anchor="center", pady=2)
        tk.Label(l2_frame, text=f"📐 Dimensões: {self.max_x}x{self.max_y} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        tk.Label(l2_frame, text=f"🔲 ROI Mín: {self.min_window_m:.2f} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        tk.Label(l2_frame, text=f"🔎 Escala Média: {avg_scale_px_m:.2f} px/m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        
        # Exibe o erro em metros (entrada) e a conversão para pixels
        self.erro_sensor_lbl = tk.Label(l2_frame, text=f"⚡ Erro Simulado: {erro_metros:.4f} m ≈ {erro_px:.2f} px", 
                                        font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280")
        self.erro_sensor_lbl.pack(side="left", padx=15)
        
    def _build_right_painel(self):
        # --- RIGHT PANEL: Metrics ---
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        # Container para colocar RMS X e RMS Y lado a lado
        rms_container = tk.Frame(self.right_frame, bg="white")
        rms_container.pack(fill="x", pady=(0, 4))
        rms_container.columnconfigure(0, weight=1)
        rms_container.columnconfigure(1, weight=1)

        # RMS X
        rms_x_frame = tk.LabelFrame(rms_container, text="RMS X (m)", 
                                   font=("Segoe UI", 9, "bold"), bg="white", fg="#333333",
                                   padx=2, pady=2, borderwidth=1, relief="solid")
        rms_x_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        rms_x_frame.pack_propagate(False)
        rms_x_frame.configure(height=140)
        
        self.rmsx_fig = Figure(figsize=(2.0, 1.0), tight_layout=True, facecolor="white")
        self.rmsx_ax = self.rmsx_fig.add_subplot(111)
        self._style_ax(self.rmsx_ax)
        self.rmsx_line, = self.rmsx_ax.plot([], [], "#333333", linewidth=2)
        self.rmsx_canvas = FigureCanvasTkAgg(self.rmsx_fig, master=rms_x_frame)
        self.rmsx_canvas.get_tk_widget().pack(fill="both", expand=True)

        # RMS Y
        rms_y_frame = tk.LabelFrame(rms_container, text="RMS Y (m)", 
                                   font=("Segoe UI", 9, "bold"), bg="white", fg="#333333",
                                   padx=2, pady=2, borderwidth=1, relief="solid")
        rms_y_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        rms_y_frame.pack_propagate(False)
        rms_y_frame.configure(height=140)
        
        self.rmsy_fig = Figure(figsize=(2.0, 1.0), tight_layout=True, facecolor="white")
        self.rmsy_ax = self.rmsy_fig.add_subplot(111)
        self._style_ax(self.rmsy_ax)
        self.rmsy_line, = self.rmsy_ax.plot([], [], "#333333", linewidth=2)
        self.rmsy_canvas = FigureCanvasTkAgg(self.rmsy_fig, master=rms_y_frame)
        self.rmsy_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Histograma Frame
        hist_frame = tk.LabelFrame(self.right_frame, text="Histograma dos Erros de Estado", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, pady=4, borderwidth=1, relief="solid")
        hist_frame.pack(fill="both", expand=True, pady=(0, 4))
        
        self.hist_fig = Figure(figsize=(4.2, 2.0), tight_layout=True, facecolor="white")
        self.hist_ax = self.hist_fig.add_subplot(111)
        self._style_ax(self.hist_ax)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Scatter Frame (Dispersão)
        scatter_frame = tk.LabelFrame(self.right_frame, text="Dispersão dos Erros (assinados)", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, pady=4, borderwidth=1, relief="solid")
        scatter_frame.pack(fill="both", expand=True, pady=(0, 0))
        
        self.scatter_fig = Figure(figsize=(4.2, 2.0), tight_layout=True, facecolor="white")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self._style_ax(self.scatter_ax)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=scatter_frame)
        self.scatter_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Store config
        self.config_Q = None
        self.config_R = None
        self.config_detector_noise = 1.0
        self.video_fps = 30.0
        self.metrics_update_counter = 0

        self.rmsx_cached = None
        self.rmsy_cached = None

    def _on_escape(self, event=None):
        try:
            self.root.state("normal")
        except Exception:
            try:
                self.root.attributes("-fullscreen", False)
            except Exception:
                pass

    def load_video(self):
        """Load a video and show first frame."""
        path = filedialog.askopenfilename(
            title="Selecione um vídeo", 
            filetypes=[("Vídeos", "*.mp4;*.mkv;*.avi;*.mov"), ("All", "*")]
        )
        if not path:
            return
        
        self.video_path = path
        self.video_path_label.config(text=os.path.basename(path))
        
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        # Get video properties
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        
        # Calculate file size
        file_size_bytes = os.path.getsize(self.video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Get file extension
        file_ext = os.path.splitext(self.video_path)[1].strip(".").upper()
        
        # Calculate total duration
        total_time = self.total_frames / self.video_fps
        total_time_str = self._format_time(total_time)
        
        cap.release()
        
        if not ret:
            messagebox.showerror("Erro", "Não foi possível ler o vídeo")
            return
        
        # Display first frame
        self.tela_viewer.display_image(frame)
        self.time_info_lbl.config(text=f"00:00 / {total_time_str}")
        
        # Update video info banner
        info_text = (f"Arquivo: {os.path.basename(path)} ({file_ext}) | "
                    f"Tamanho: {file_size_mb:.1f} MB | "
                    f"FPS: {self.video_fps:.1f} | "
                    f"Frames: {self.total_frames} | "
                    f"Taxa: {width}×{height} ({total_time_str})")
        self.video_info_lbl.config(text=info_text)
        
        # Enable EXEC button
        self.exec_btn.config(state="normal")
        self.status_lbl.config(text="Status: Pronto para executar")

    def execute_processing(self):
        """Process video and save to src/data/"""
        if not self.video_path or self.processing:
            return
        
        try:
            self._parse_config()
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro nas configurações: {error_msg}"))
            self.processing = False
            return  # Para a execução IMEDIATAMENTE se houver erro!
        
        # Inicia o processamento apenas se o try acima deu certo
        self.processing = True
        self.exec_btn.config(state="disabled")
        self.load_btn.config(state="disabled")
        self.status_lbl.config(text="Status: Processando...")
        
        self.worker = threading.Thread(target=self._process_video, daemon=True)
        self.worker.start()

    def _style_ax(self, ax):
        """Aplica o estilo visual limpo para os eixos do Matplotlib."""
        ax.set_facecolor("#f5f5f5")
        ax.tick_params(colors="#666666", labelsize=8)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _parse_config(self):
        """Parse Q, R, detector noise, e localização das Torres da UI.
        O erro do detector é lido em metros e convertido para pixels."""
        import numpy as np

        # 1. Parse Q from individual entries
        q_vals = []
        entries_q = getattr(self, 'q_entries', [])
        if entries_q:
            for entry in entries_q:
                if entry is not None:
                    try: 
                        val_str = entry.get().strip()
                        q_vals.append(float(val_str))
                    except (ValueError, AttributeError): 
                        q_vals.append(1e-2)
                else:
                    q_vals.append(1e-2)
        
        while len(q_vals) < 6:
            q_vals.append(1e-2)
        
        # 2. Parse R from individual entries
        r_vals = []
        entries_r = getattr(self, 'r_entries', [])
        if entries_r:
            for entry in entries_r:
                if entry is not None:
                    try: 
                        val_str = entry.get().strip()
                        r_vals.append(float(val_str))
                    except (ValueError, AttributeError): 
                        r_vals.append(1e-1)
                else:
                    r_vals.append(1e-1)

        while len(r_vals) < 2:
            r_vals.append(1e-1)

        # 3. Parse Torres
        towers_temp = []
        if hasattr(self, 'tower_entries') and self.tower_entries is not None:
            for pair in self.tower_entries:
                if pair is not None and len(pair) == 2:
                    ent_x, ent_y = pair
                    try:
                        x = float(ent_x.get().strip()) if ent_x is not None else 0.0
                        y = float(ent_y.get().strip()) if ent_y is not None else 0.0
                        towers_temp.append([x, y])
                    except (ValueError, AttributeError):
                        towers_temp.append([0.0, 0.0])
            
            if towers_temp:
                self.towers = np.array(towers_temp)
            
        # 4. Obtém o erro digitado na interface em METROS
        erro_metros = 1.0
        if hasattr(self, 'detector_noise_entry') and self.detector_noise_entry is not None:
            try:
                erro_metros = float(self.detector_noise_entry.get().strip())
            except (ValueError, AttributeError):
                erro_metros = 1.0

        # Proteções para dimensões físicas e de pixel
        max_x = self.max_x if hasattr(self, 'max_x') and self.max_x is not None and self.max_x > 0 else 1.0
        max_y = self.max_y if hasattr(self, 'max_y') and self.max_y is not None and self.max_y > 0 else 1.0
        v_width = self.video_width if hasattr(self, 'video_width') and self.video_width is not None and self.video_width > 0 else 640
        v_height = self.video_height if hasattr(self, 'video_height') and self.video_height is not None and self.video_height > 0 else 480

        px_m_x = v_width / max_x
        px_m_y = v_height / max_y
        
        # 5. Salva o ruído nas DUAS métricas
        self.config_detector_noise = px_m_x * erro_metros   
        self.config_detector_noise_m = erro_metros          

        # 6. Atualiza a label da dashboard
        if hasattr(self, 'erro_sensor_lbl') and self.erro_sensor_lbl is not None:
            avg_scale_px_m = (px_m_x + px_m_y) / 2.0
            erro_px = erro_metros * avg_scale_px_m
            try:
                self.erro_sensor_lbl.config(text=f"⚡ Erro Simulado: {erro_metros:.4f} m ≈ {erro_px:.2f} px")
            except Exception:
                pass

        # 7. Ajuste Inicial/Reset das informações na Seção de Status e Métricas
        if hasattr(self, 'status_lbl') and self.status_lbl is not None:
            try: self.status_lbl.config(text="Status: Aguardando Ação", fg="#16a34a")
            except Exception: pass
            
        if hasattr(self, 'det_rate_lbl') and self.det_rate_lbl is not None:
            try: self.det_rate_lbl.config(text="Taxa de Detecção: --%", fg="#4b5563")
            except Exception: pass
            
        if hasattr(self, 'inlier_rate_lbl') and self.inlier_rate_lbl is not None:
            try: self.inlier_rate_lbl.config(text="Inliers (na Janela): --%", fg="#4b5563")
            except Exception: pass
            
        if hasattr(self, 'rmse_lbl') and self.rmse_lbl is not None:
            try: self.rmse_lbl.config(text="RMSE (X | Y): -- | -- m", fg="#4b5563")
            except Exception: pass
            
        if hasattr(self, 'mean_err_lbl') and self.mean_err_lbl is not None:
            try: self.mean_err_lbl.config(text="Erro Médio (X | Y): -- | -- m", fg="#4b5563")
            except Exception: pass
            
        if hasattr(self, 'nis_lbl') and self.nis_lbl is not None:
            try: self.nis_lbl.config(text="NIS Médio: --", fg="#4b5563")
            except Exception: pass

        # ADICIONADO: Limpa a label de Regime Estacionário
        if hasattr(self, 'steady_state_lbl') and self.steady_state_lbl is not None:
            try: self.steady_state_lbl.config(text="Regime Estacionário: -- s (-- frames)", fg="#4b5563")
            except Exception: pass
            
        # Cria ou reseta as variáveis internas utilizadas no processamento
        self.current_inliers = 0.0
        self.current_detected = 0.0
        
        # ADICIONADO: Variáveis de estado do Filtro para convergência
        self.steady_state_reached = False
        self.steady_state_frame = -1

        # 8. Salva as configurações das matrizes
        self.config_Q = q_vals[:6]  
        self.config_R = r_vals[:2]

    def _process_video(self):
        """Process video with Kalman filter and save to src/data/."""
        cap = None 
        out = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.fps = fps 
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            os.makedirs("FiltroKalman/src/data", exist_ok=True)
            output_path = "FiltroKalman/src/data/output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            self.world_m = World(dimPX=w, dimPY=h, dimPOX=self.max_x, dimPOY=self.max_y)
            dt = 1.0 / fps
            
            q_vals = self.config_Q if len(self.config_Q) >= 6 else self.config_Q + [1e-1] * (6 - len(self.config_Q))
            r_vals = self.config_R[:2] if len(self.config_R) >= 2 else self.config_R + [1e-1]

            kf = Entidy(dt=dt, q_diag=q_vals[:6], r_diag=r_vals[:2])
            
            self.saved_Q = kf.Q if hasattr(kf, 'Q') else np.diag(q_vals[:6])
            self.saved_R = kf.R if hasattr(kf, 'R') else np.diag(r_vals[:2])
            self.saved_Qd = getattr(kf, 'Qd', getattr(kf, 'Q_discrete', getattr(kf, 'Q_d', None)))

            frame_count = 0
            frames_with_meas = 0        
            self.meas_inside_roi = 0         

            self.meas_pts = []
            self.filt_pts = []
            self.sqerr_x = []
            self.sqerr_y = []
            self.kalman_windows = []
            self.nis_vals = []
            self.innov_x = []            
            self.innov_y = []            

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Para uma lógica de Visão Ativa completa no futuro, 
                # o detect_centroid poderia receber o 'est_m' do frame anterior aqui.
                meas_px = detect_centroid(frame, noise_std=self.config_detector_noise)
                meas_m = self.world_m.img2world(meas_px[0], meas_px[1]) if meas_px is not None else None
                
                # 1. PREDIÇÃO (A Priori)
                kf.predict()
                pos_pred = kf.get_position()  
                P_pred = kf.P.copy() if hasattr(kf, 'P') else None

                # A incerteza (S) teórica continua sendo calculada via Predição
                if P_pred is not None:
                    S = P_pred[:2, :2] + np.diag(r_vals[:2])
                    std_innov_x_m = max(self.min_window_m, np.sqrt(S[0, 0]) * 3)
                    std_innov_y_m = max(self.min_window_m, np.sqrt(S[1, 1]) * 3)
                else:
                    std_innov_x_m = self.min_window_m
                    std_innov_y_m = self.min_window_m

                if meas_m is not None:
                    # Inovação a priori (Apenas para fins estatísticos como o NIS)
                    innov = np.array([meas_m[0] - pos_pred[0], meas_m[1] - pos_pred[1]])
                    self.innov_x.append(innov[0])
                    self.innov_y.append(innov[1])

                    if P_pred is not None:
                        try:
                            invS = np.linalg.inv(S)
                            nis = innov @ invS @ innov
                        except np.linalg.LinAlgError:
                            nis = np.nan
                    else:
                        nis = np.nan
                        
                    self.nis_vals.append(nis)

                    # 2. ATUALIZAÇÃO (A Posteriori)
                    kf.update(meas_m)
                    frames_with_meas += 1

                else:
                    self.innov_x.append(np.nan)
                    self.innov_y.append(np.nan)
                    self.nis_vals.append(np.nan)

                # 3. GUARDAR ESTADOS FINAIS (A Posteriori)
                est_m = kf.get_position()
                P_mat = kf.P.copy() if hasattr(kf, 'P') else None

                # --- LÓGICA DO ROI E AVALIAÇÃO DA PREVISÃO (A POSTERIORI) ---
                if est_m is not None and meas_m is not None:
                    innov_posteriori = np.array([meas_m[0] - est_m[0], meas_m[1] - est_m[1]])

                    dist_x = abs(innov_posteriori[0])
                    dist_y = abs(innov_posteriori[1])

                    # Consideramos "Inlier" se a medição ficou dentro da janela de incerteza da estimativa final
                    if dist_x <= std_innov_x_m and dist_y <= std_innov_y_m:
                        self.meas_inside_roi += 1

                # CÁLCULO DO REGIME ESTACIONÁRIO 
                if not getattr(self, 'steady_state_reached', False) and P_mat is not None and len(self.kalman_windows) > 0:
                    P_prev = self.kalman_windows[-1]
                    if P_prev is not None:
                        variacao_P = np.trace(np.abs(P_mat - P_prev))
                        if variacao_P < 1e-2:
                            self.steady_state_reached = True
                            self.steady_state_frame = frame_count
                            tempo_segundos = frame_count / self.fps
                            
                            self.root.after(0, lambda f=frame_count, t=tempo_segundos: 
                                self.steady_state_lbl.config(
                                    text=f"Regime Estacionário: {t:.2f} s ({f} frames)", 
                                    fg="#1e3a8a", font=("Segoe UI", 8, "bold")
                                ) if hasattr(self, 'steady_state_lbl') else None
                            )
                
                self.kalman_windows.append(P_mat)

                if meas_m is None:
                    self.meas_pts.append(None)
                    self.sqerr_x.append(np.nan)
                    self.sqerr_y.append(np.nan)
                else:
                    mx, my = float(meas_m[0]), float(meas_m[1])
                    ex, ey = float(est_m[0]), float(est_m[1])
                    self.meas_pts.append((mx, my))
                    dx, dy = ex - mx, ey - my
                    self.sqerr_x.append(dx ** 2)
                    self.sqerr_y.append(dy ** 2)

                self.filt_pts.append((ex, ey))

                # --- RENDERIZAÇÃO GRÁFICA ---
                ann = frame.copy()
                if self.show_traj.get() and len(self.filt_pts) >= 2:
                    filt_poly = [self.world_m.world2img(p[0], p[1]) for p in self.filt_pts if p is not None]
                    if len(filt_poly) >= 2:
                        cv2.polylines(ann, [np.array(filt_poly, dtype=np.int32)], False, YELLOW_COLOR, 2)

                if self.show_detect.get() and len(self.meas_pts) >= 2:
                    meas_poly = [self.world_m.world2img(p[0], p[1]) for p in self.meas_pts if p is not None]
                    if len(meas_poly) >= 2:
                        cv2.polylines(ann, [np.array(meas_poly, dtype=np.int32)], False, GREEN_COLOR, 1)

                # DESENHO DA JANELA ANCORADA NO ESTADO A POSTERIORI (est_m)
                if self.show_window.get() and est_m is not None:
                    px1, py1 = self.world_m.world2img(est_m[0] - std_innov_x_m, est_m[1] + std_innov_y_m)
                    px2, py2 = self.world_m.world2img(est_m[0] + std_innov_x_m, est_m[1] - std_innov_y_m)
                    cv2.rectangle(ann, 
                                  (max(0, min(w, int(px1))), max(0, min(h, int(py1)))),
                                  (max(0, min(w, int(px2))), max(0, min(h, int(py2)))),
                                  (0, 255, 0), 2)
                    cx, cy = self.world_m.world2img(est_m[0], est_m[1])
                    cv2.circle(ann, (int(cx), int(cy)), 2, (0, 255, 0), -1)

                est_px = self.world_m.world2img(est_m[0], est_m[1])
                if self.show_kalman.get():
                    cv2.circle(ann, (int(est_px[0]), int(est_px[1])), 6, BLUE_COLOR, -1)

                if self.show_detect.get() and meas_m is not None:
                    valid_px = self.world_m.world2img(meas_m[0], meas_m[1])
                    cv2.circle(ann, (int(valid_px[0]), int(valid_px[1])), 6, (255, 0, 0), -1)

                out.write(ann)
                frame_count += 1

            # ========== CÁLCULO CENTRALIZADO DE MÉTRICAS ==========
            self.total_frames = frame_count
            self.detection_rate = (frames_with_meas / frame_count * 100.0) if frame_count > 0 else 0.0
            self.inlier_rate = (self.meas_inside_roi / frames_with_meas * 100.0) if frames_with_meas > 0 else 0.0
            
            self.rmse_x_total = np.sqrt(np.nanmean(self.sqerr_x)) if self.sqerr_x else 0.0
            self.rmse_y_total = np.sqrt(np.nanmean(self.sqerr_y)) if self.sqerr_y else 0.0
            
            signed_dx = []
            signed_dy = []
            for m_pt, f_pt in zip(self.meas_pts, self.filt_pts):
                if m_pt is not None:
                    signed_dx.append(f_pt[0] - m_pt[0])
                    signed_dy.append(f_pt[1] - m_pt[1])
                    
            self.mean_signed_dx = np.mean(signed_dx) if signed_dx else 0.0
            self.mean_signed_dy = np.mean(signed_dy) if signed_dy else 0.0

            valid_nis = [n for n in self.nis_vals if not np.isnan(n)]
            self.mean_nis = np.mean(valid_nis) if valid_nis else 0.0

            self.sensor_detection_rate = self.detection_rate
            self.roi_accuracy_rate = self.inlier_rate
            self.processed_video_path = output_path

            self.root.after(0, self._update_ui_metrics_and_complete)

        except Exception as e:
            errorMsg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao processar: {errorMsg}"))
        finally:
            self.processing = False
            if cap is not None: cap.release()
            if out is not None: out.release()
    def _update_ui_metrics_and_complete(self):
        """Atualiza dinamicamente as labels do painel esquerdo e executa callbacks finais."""
        # Status
        self.status_lbl.config(text="Status: Processado", fg="#16a34a")
        
        # Taxas originais
        self.det_rate_lbl.config(text=f"Taxa de Detecção: {self.detection_rate:.1f}%")
        self.inlier_rate_lbl.config(text=f"Inliers (na Janela): {self.inlier_rate:.1f}%")
        
        # Métricas estatísticas de erro
        self.rmse_lbl.config(text=f"RMSE (X | Y): {self.rmse_x_total:.4f} | {self.rmse_y_total:.4f} m")
        
        # Atualização do Erro Médio (usando :+.4f para forçar a exibição do sinal + ou -)
        self.mean_err_lbl.config(text=f"Erro Médio (X | Y): {self.mean_signed_dx:+.4f} | {self.mean_signed_dy:+.4f} m")
        
        # Consistência
        self.nis_lbl.config(text=f"NIS Médio: {self.mean_nis:.3f} (Ideal ~2)")
        
        # Callback final original do seu sistema
        self._on_processing_complete()

    def _process_robot_simulation(self):
        """Gera a Arena Sintética com EKF e grava como output.mp4 sem precisar de vídeos prévios."""
        out = None
        try:
            fps = 30.0
            self.fps = fps
            w, h = self.video_width, self.video_height
            
            os.makedirs("FiltroKalman/src/data", exist_ok=True)
            output_path = "FiltroKalman/src/data/output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            self.world_m = World(dimPX=w, dimPY=h, dimPOX=self.max_x, dimPOY=self.max_y)
            dt = 1.0 / fps
            
            q_vals = self.config_Q if len(self.config_Q) >= 6 else self.config_Q + [1e-1] * (6 - len(self.config_Q))
            r_vals = self.config_R[:2] if len(self.config_R) >= 2 else self.config_R + [1e-1]

            # Instanciação da Entidade Robótica (EKF)
            kf = Robot(dt=dt, q_diag=q_vals[:6], r_diag=r_vals[:2])
            
            # Inicialização de Variáveis e Arrays de Métricas
            total_frames_sim = 450 # Exemplo: 15 segundos de simulação
            self.total_frames = total_frames_sim
            frame_count = 0
            frames_with_meas = 0        
            self.meas_inside_roi = 0         
            self.meas_pts, self.filt_pts, self.sqerr_x, self.sqerr_y = [], [], [], []
            self.kalman_windows, self.nis_vals, self.innov_x, self.innov_y = [], [], [], []

            for frame_idx in range(total_frames_sim):
                # 1. Cria Fundo Escuro para a Arena Cartesiana
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 2. Desenha as Torres Configuradas e salvas no parse_config
                if hasattr(self, 'towers') and self.towers is not None:
                    for i, (tx, ty) in enumerate(self.towers):
                        tpx, tpy = self.world_m.world2img(tx, ty)
                        cv2.circle(frame, (int(tpx), int(tpy)), 10, (200, 200, 200), 2)
                        cv2.putText(frame, f"T{i+1}", (int(tpx)-12, int(tpy)-15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # =================================================================
                # INTEGRAÇÃO DO FUTURO GERADOR DE TRAJETÓRIAS:
                # x_true, y_true = meu_gerador.gerar(frame_idx, mode="laminescata")
                # =================================================================
                # Movimento Circular Provisório:
                angulo = frame_idx * (2 * np.pi / total_frames_sim)
                x_true = (self.max_x / 2.0) + (self.max_x / 3.0) * np.cos(angulo)
                y_true = (self.max_y / 2.0) + (self.max_y / 3.0) * np.sin(angulo)
                
                # Renderiza a posição Verdadeira do Robô (Ground Truth) em Branco
                gt_px = self.world_m.world2img(x_true, y_true)
                cv2.circle(frame, (int(gt_px[0]), int(gt_px[1])), 6, (255, 255, 255), -1)

                # 3. Medição Simulada pelo EKF
                meas_px = detect_bot(frame, noise_std=self.config_detector_noise_m, towers=self.towers)
                meas_m = self.world_m.img2world(meas_px[0], meas_px[1]) if meas_px is not None else None
                
                # 4. PREDIÇÃO EKF
                kf.predict()
                pos_pred = kf.get_position()  
                P_pred = kf.P.copy() if hasattr(kf, 'P') else None

                if P_pred is not None:
                    S = P_pred[:2, :2] + np.diag(r_vals[:2])
                    std_innov_x_m = max(self.min_window_m, np.sqrt(S[0, 0]) * 3)
                    std_innov_y_m = max(self.min_window_m, np.sqrt(S[1, 1]) * 3)
                else:
                    std_innov_x_m = self.min_window_m
                    std_innov_y_m = self.min_window_m

                if meas_m is not None:
                    innov = np.array([meas_m[0] - pos_pred[0], meas_m[1] - pos_pred[1]])
                    self.innov_x.append(innov[0])
                    self.innov_y.append(innov[1])
                    if P_pred is not None:
                        try:
                            nis = innov @ np.linalg.inv(S) @ innov
                        except np.linalg.LinAlgError:
                            nis = np.nan
                    else: nis = np.nan
                        
                    self.nis_vals.append(nis)

                    # Gating
                    if abs(innov[0]) <= std_innov_x_m and abs(innov[1]) <= std_innov_y_m:
                        self.meas_inside_roi += 1
                        
                    # ATUALIZAÇÃO EKF
                    kf.update(meas_m)
                    frames_with_meas += 1
                else:
                    self.innov_x.append(np.nan)
                    self.innov_y.append(np.nan)
                    self.nis_vals.append(np.nan)

                # 5. ANÁLISE DE ERROS BASEADA NO "GROUND TRUTH" E DESENHO
                est_m = kf.get_position()
                self.kalman_windows.append(P_pred) 

                if meas_m is None:
                    self.meas_pts.append(None)
                    self.sqerr_x.append(np.nan)
                    self.sqerr_y.append(np.nan)
                else:
                    self.meas_pts.append((float(meas_m[0]), float(meas_m[1])))
                    # VANTAGEM DE EKF SINTÉTICO: O erro matemático real é calculado
                    # contra a posição do gerador (x_true, y_true), e não contra a medição viciada!
                    self.sqerr_x.append((est_m[0] - x_true) ** 2)
                    self.sqerr_y.append((est_m[1] - y_true) ** 2)

                self.filt_pts.append((float(est_m[0]), float(est_m[1])))

                # Desenhos (Copiam integralmente o seu _process_video normal)
                ann = frame.copy()
                if self.show_traj_val and len(self.filt_pts) >= 2:
                    filt_poly = [self.world_m.world2img(p[0], p[1]) for p in self.filt_pts if p is not None]
                    if len(filt_poly) >= 2: cv2.polylines(ann, [np.array(filt_poly, dtype=np.int32)], False, GREEN_COLOR, 2)

                if self.show_detect_val and len(self.meas_pts) >= 2:
                    meas_poly = [self.world_m.world2img(p[0], p[1]) for p in self.meas_pts if p is not None]
                    if len(meas_poly) >= 2: cv2.polylines(ann, [np.array(meas_poly, dtype=np.int32)], False, RED_COLOR, 1)

                if self.show_window_val and pos_pred is not None:
                    px1, py1 = self.world_m.world2img(pos_pred[0] - std_innov_x_m, pos_pred[1] + std_innov_y_m)
                    px2, py2 = self.world_m.world2img(pos_pred[0] + std_innov_x_m, pos_pred[1] - std_innov_y_m)
                    cv2.rectangle(ann, (max(0, min(w, int(px1))), max(0, min(h, int(py1)))), (max(0, min(w, int(px2))), max(0, min(h, int(py2)))), YELLOW_COLOR, 2)
                    cx, cy = self.world_m.world2img(pos_pred[0], pos_pred[1])
                    cv2.circle(ann, (int(cx), int(cy)), 2, YELLOW_COLOR, -1)

                if self.show_detect_val and meas_m is not None:
                    valid_px = self.world_m.world2img(meas_m[0], meas_m[1])
                    cv2.circle(ann, (int(valid_px[0]), int(valid_px[1])), 6, RED_COLOR, -1)

                if self.show_kalman_val:
                    est_px = self.world_m.world2img(est_m[0], est_m[1])
                    cv2.circle(ann, (int(est_px[0]), int(est_px[1])), 4, BLUE_COLOR, -1)

                out.write(ann)
                frame_count += 1

            # Finalização Segura
            self.total_frames = frame_count
            self.detection_rate = (frames_with_meas / frame_count * 100.0) if frame_count > 0 else 0.0
            self.inlier_rate = (self.meas_inside_roi / frames_with_meas * 100.0) if frames_with_meas > 0 else 0.0
            
            self.processed_video_path = output_path
            self.root.after(0, self._on_processing_complete)
            
        except Exception as e:
            errorMsg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro na Simulação EKF", f"Erro: {errorMsg}"))
        finally:
            self.processing = False
            if out is not None: out.release()

    def _find_convergence_frame(self, running_rms, final_val, tol=0.05, min_stable=10):
        """
        Retorna o índice (em frames válidos) em que o running RMS fica consistentemente
        <= (final_val * (1 + tol)) por pelo menos min_stable pontos consecutivos.
        """
        if final_val is None or len(running_rms) < min_stable:
            return None
        threshold = final_val * (1 + tol)
        start = max(1, len(running_rms) // 5)  # ignora os primeiros 20%
        for i in range(start, len(running_rms) - min_stable + 1):
            if all(v <= threshold for v in running_rms[i:i+min_stable]):
                return i
        return None

    def save_results(self):
        """Salva gráficos detalhados, relatório e dados brutos em CSV em src/results/"""
        if not self.filt_pts or not self.meas_pts:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar.")
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "results"
            save_dir = f"FiltroKalman/src/results/{video_name}"
            os.makedirs(save_dir, exist_ok=True)

            # ========== Preparação de dados ==========
            signed_dx = []
            signed_dy = []
            for m_pt, f_pt in zip(self.meas_pts, self.filt_pts):
                if m_pt is not None:
                    signed_dx.append(f_pt[0] - m_pt[0])
                    signed_dy.append(f_pt[1] - m_pt[1])

            valid_nis = [n for n in self.nis_vals if not np.isnan(n)]

            sx = np.array([v for v in self.sqerr_x if not np.isnan(v)])
            sy = np.array([v for v in self.sqerr_y if not np.isnan(v)])
            run_rms_x = np.sqrt(np.cumsum(sx) / np.arange(1, sx.size + 1)) if sx.size > 0 else []
            run_rms_y = np.sqrt(np.cumsum(sy) / np.arange(1, sy.size + 1)) if sy.size > 0 else []

            # ===== GRÁFICO 1: TRAJETÓRIA =====
            fig1 = Figure(figsize=(12, 8), tight_layout=True, dpi=150)
            ax1 = fig1.add_subplot(111)
            xs_meas = [p[0] for p in self.meas_pts if p is not None]
            ys_meas = [p[1] for p in self.meas_pts if p is not None]
            xs_filt = [p[0] for p in self.filt_pts]
            ys_filt = [p[1] for p in self.filt_pts]
            ax1.plot(xs_meas, ys_meas, "r.-", label="Medido (Detector)", linewidth=1, markersize=4, alpha=0.5)
            ax1.plot(xs_filt, ys_filt, "g-", label="Kalman Filtrado", linewidth=2.5, alpha=0.9)
            ax1.set_xlabel("Posição X (metros)", fontweight="bold")
            ax1.set_ylabel("Posição Y (metros)", fontweight="bold")
            ax1.set_title(f"Trajetória Espacial | Detecções: {self.detection_rate:.1f}% | Inliers: {self.inlier_rate:.1f}%", fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle="--")
            fig1.savefig(f"{save_dir}/{video_name}_1_traj.png")

            # ===== GRÁFICO 2: RMS ACUMULADO + CONVERGÊNCIA =====
            fig2 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax2 = fig2.add_subplot(111)
            if len(run_rms_x) > 0:
                ax2.plot(run_rms_x, label="RMS X", color="blue")
                ax2.plot(run_rms_y, label="RMS Y", color="orange")
                final_x = run_rms_x[-1] if len(run_rms_x) > 0 else None
                final_y = run_rms_y[-1] if len(run_rms_y) > 0 else None
                conv_idx_x = self._find_convergence_frame(run_rms_x, final_x)
                conv_idx_y = self._find_convergence_frame(run_rms_y, final_y)
                if conv_idx_x is not None:
                    ax2.axvline(conv_idx_x, color='blue', linestyle=':', alpha=0.7,
                                label=f"Conv. X: {conv_idx_x/self.fps:.2f}s")
                if conv_idx_y is not None:
                    ax2.axvline(conv_idx_y, color='orange', linestyle=':', alpha=0.7,
                                label=f"Conv. Y: {conv_idx_y/self.fps:.2f}s")
            ax2.set_xlabel("Frames", fontweight="bold")
            ax2.set_ylabel("Erro RMS Acumulado (metros)", fontweight="bold")
            ax2.set_title("Evolução do Erro RMS", fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3, linestyle="--")
            fig2.savefig(f"{save_dir}/{video_name}_2_rms.png")

            # ===== GRÁFICO 3: DISPERSÃO (ERROS ASSINADOS) =====
            fig3 = Figure(figsize=(8, 8), tight_layout=True, dpi=150)
            ax3 = fig3.add_subplot(111)
            ax3.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=20)
            ax3.axhline(0, color='black', linewidth=1)
            ax3.axvline(0, color='black', linewidth=1)
            ax3.set_xlabel("Erro X (m)", fontweight="bold")
            ax3.set_ylabel("Erro Y (m)", fontweight="bold")
            ax3.set_title("Dispersão dos Erros de Estimação (assinados)", fontweight="bold")
            ax3.grid(True, alpha=0.3, linestyle="--")
            fig3.savefig(f"{save_dir}/{video_name}_3_scatter.png")

            # ===== GRÁFICO 4: NIS =====
            fig4 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax4 = fig4.add_subplot(111)
            frames_nis = [i for i, n in enumerate(self.nis_vals) if not np.isnan(n)]
            if valid_nis:
                ax4.plot(frames_nis, valid_nis, 'm-', alpha=0.7, label="NIS Calculado")
                ax4.axhline(5.99, color='r', linestyle='--', linewidth=2, label="Limite 95% Confiança (χ²)")
                ax4.set_ylim(0, max(15, float(np.percentile(valid_nis, 95) * 1.5)))
            ax4.set_xlabel("Frames", fontweight="bold")
            ax4.set_ylabel("Valor NIS", fontweight="bold")
            ax4.set_title("Teste de Consistência NIS (covariância corrigida)", fontweight="bold")
            ax4.legend()
            ax4.grid(True, alpha=0.3, linestyle="--")
            fig4.savefig(f"{save_dir}/{video_name}_4_nis.png")

            # ===== GRÁFICO 5: HISTOGRAMA DOS ERROS =====
            fig5 = Figure(figsize=(10, 6), tight_layout=True, dpi=150)
            ax5 = fig5.add_subplot(111)
            ax5.hist(signed_dx, bins=30, alpha=0.5, color='blue', label='Erros X (m)')
            ax5.hist(signed_dy, bins=30, alpha=0.5, color='orange', label='Erros Y (m)')
            ax5.set_xlabel("Erro (metros)", fontweight="bold")
            ax5.set_ylabel("Frequência", fontweight="bold")
            ax5.set_title("Histograma dos Erros de Estado (assinados)", fontweight="bold")
            ax5.legend()
            ax5.grid(True, alpha=0.3, linestyle="--")
            fig5.savefig(f"{save_dir}/{video_name}_5_hist.png")

            plt.close('all')

            # ========== CÁLCULO DE MÉTRICAS COMPLEMENTARES ==========
            # Usando as variáveis que já calculamos no _process_video para consistência:
            rmse_x = getattr(self, 'rmse_x_total', 0.0)
            rmse_y = getattr(self, 'rmse_y_total', 0.0)
            mean_dx = getattr(self, 'mean_signed_dx', 0.0)
            mean_dy = getattr(self, 'mean_signed_dy', 0.0)
            mean_nis_val = getattr(self, 'mean_nis', 0.0)

            # Métricas que só interessam para o relatório:
            std_dx = np.std(signed_dx) if signed_dx else 0.0
            std_dy = np.std(signed_dy) if signed_dy else 0.0
            max_err_x = np.max(np.abs(signed_dx)) if signed_dx else 0.0
            max_err_y = np.max(np.abs(signed_dy)) if signed_dy else 0.0

            nis_above_95 = sum(1 for n in valid_nis if n > 5.99)
            nis_pct_above = (nis_above_95 / len(valid_nis)) * 100 if valid_nis else 0.0

            conv_idx_x = self._find_convergence_frame(run_rms_x, run_rms_x[-1] if len(run_rms_x) else None)
            conv_idx_y = self._find_convergence_frame(run_rms_y, run_rms_y[-1] if len(run_rms_y) else None)
            conv_time_x = conv_idx_x / self.fps if conv_idx_x is not None else None
            conv_time_y = conv_idx_y / self.fps if conv_idx_y is not None else None

            # ========== RELATÓRIO TXT ==========
            txt_path = f"{save_dir}/{video_name}_metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("====================================================\n")
                f.write("      RESUMO DE MÉTRICAS - FILTRO DE KALMAN         \n")
                f.write("====================================================\n\n")
                f.write(f"Arquivo Fonte: {self.video_path}\n")
                f.write(f"Data da Análise: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Total de Frames: {self.total_frames}\n")
                f.write(f"FPS: {self.fps:.2f}\n\n")

                fmt_opts = {'precision': 6, 'suppress_small': True, 'separator': '  '}
                f.write("--- PARÂMETROS DO FILTRO ---\n")
                if hasattr(self, 'saved_Q') and self.saved_Q is not None:
                    f.write("Matriz de Ruído de Processo Contínuo (Q):\n")
                    f.write(f"{np.array2string(np.array(self.saved_Q), **fmt_opts)}\n\n")
                if hasattr(self, 'saved_Qd') and self.saved_Qd is not None:
                    f.write("Matriz de Ruído de Processo Discretizada (Qd):\n")
                    f.write(f"{np.array2string(np.array(self.saved_Qd), **fmt_opts)}\n\n")
                else:
                    f.write("Matriz de Ruído de Processo Discretizada (Qd): [Não Disponível]\n\n")
                if hasattr(self, 'saved_R') and self.saved_R is not None:
                    f.write("Matriz de Ruído de Medição (R):\n")
                    f.write(f"{np.array2string(np.array(self.saved_R), **fmt_opts)}\n\n")
                f.write("----------------------------------------------------\n\n")

                f.write("--- TAXAS DE DETECÇÃO E CONSISTÊNCIA ---\n")
                f.write(f"Taxa de Detecção (frames com medição): {self.detection_rate:.2f}%\n")
                f.write(f"Taxa de Inliers (medições dentro da janela 3σ): {self.inlier_rate:.2f}%\n\n")

                f.write("--- ERROS DE ESTIMAÇÃO (METROS) ---\n")
                f.write(f"RMSE X: {rmse_x:.4f} m\n")
                f.write(f"RMSE Y: {rmse_y:.4f} m\n")
                f.write(f"Erro Médio (viés) em X: {mean_dx:+.4f} m\n")
                f.write(f"Erro Médio (viés) em Y: {mean_dy:+.4f} m\n")
                f.write(f"Desvio Padrão Erro X: {std_dx:.4f} m\n")
                f.write(f"Desvio Padrão Erro Y: {std_dy:.4f} m\n")
                f.write(f"Erro Máximo Absoluto X: {max_err_x:.4f} m\n")
                f.write(f"Erro Máximo Absoluto Y: {max_err_y:.4f} m\n\n")

                f.write("--- CONVERGÊNCIA DO RMS ---\n")
                if conv_time_x is not None:
                    f.write(f"RMS X convergiu em {conv_idx_x} frames ({conv_time_x:.2f} s)\n")
                else:
                    f.write("RMS X não atingiu convergência dentro do vídeo.\n")
                if conv_time_y is not None:
                    f.write(f"RMS Y convergiu em {conv_idx_y} frames ({conv_time_y:.2f} s)\n")
                else:
                    f.write("RMS Y não atingiu convergência dentro do vídeo.\n")
                f.write("(Critério: erro RMS ≤ 5% do valor final por 10 frames consecutivos)\n\n")

                f.write("--- AVALIAÇÃO DE CONSISTÊNCIA (NIS) ---\n")
                f.write(f"NIS Médio (ideal ≈ 2): {mean_nis_val:.4f}\n")
                f.write(f"Percentual acima do limite 95% (5.99): {nis_pct_above:.2f}%\n")
                f.write(" * Nota: O NIS avalia se a covariância reflete a real incerteza do modelo.\n")
                f.write("   Uma porcentagem acima de ~5% no limite indica que o filtro está subestimando\n")
                f.write("   o ruído ou divergindo levemente.\n\n")
                f.write("====================================================\n")

            # ========== CSV ==========
            csv_path = f"{save_dir}/{video_name}_positions.csv"
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv, delimiter=',')
                writer.writerow(["Frame", "Meas_X(m)", "Meas_Y(m)", "Filt_X(m)", "Filt_Y(m)"])
                for frame_idx, (m_pt, f_pt) in enumerate(zip(self.meas_pts, self.filt_pts)):
                    mx = f"{m_pt[0]:.4f}" if m_pt is not None else ""
                    my = f"{m_pt[1]:.4f}" if m_pt is not None else ""
                    fx = f"{f_pt[0]:.4f}" if f_pt is not None else ""
                    fy = f"{f_pt[1]:.4f}" if f_pt is not None else ""
                    writer.writerow([frame_idx, mx, my, fx, fy])

            messagebox.showinfo("Sucesso", f"Análise completa e arquivo .csv salvos em:\n{save_dir}/")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao gerar relatórios: {error_msg}"))
        finally:
            self.processing = False
            
    def _on_processing_complete(self):
        """Seletor de interface chamado ao finalizar o processamento."""
        self.status_lbl.config(text=f"Status: Concluído | Detecção: {self.inlier_rate:.1f}%")
        self.exec_btn.config(state="normal")
        self.load_btn.config(state="normal")

        if self.processed_video_path:
            self.cap = cv2.VideoCapture(self.processed_video_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0

            ret, frame = self.cap.read()
            if ret:
                self.tela_viewer.display_image(frame)
                total_time = self.total_frames / self.video_fps
                self.time_info_lbl.config(text=f"00:00 / {self._format_time(total_time)}")

            self.prev_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.save_btn.config(state="normal")

            self._update_metrics_plots()
    def toggle_playback(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "⏵")

    def next_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()

    def prev_frame(self):
        if not self.cap:
            return
        self.playing = False
        self.play_btn.config(text="⏵")
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._display_current_frame()

    def _display_current_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.tela_viewer.display_image(frame)
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            current_str = self._format_time(self.current_frame_idx / fps)
            total_str = self._format_time(self.total_frames / fps)
            self.time_info_lbl.config(text=f"{current_str} / {total_str}")
            self._update_metrics_plots()

    def _poll_playback(self):
        if self.playing and self.cap and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()
        elif self.playing and self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="⏵")
        self.root.after(33, self._poll_playback)
    
    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def _update_metrics_plots(self):
        """Atualiza dinamicamente as curvas de telemetria baseando-se no espaço métrico (m)."""
        self.metrics_update_counter += 1
        if self.metrics_update_counter % 3 != 0:  
            return
        
        current_idx = self.current_frame_idx + 1
        
        # --- RMS X e Y (mantidos como antes) ---
        sx = np.array([v for v in self.sqerr_x[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sx.size > 0:
            running_mean_x = np.cumsum(sx) / np.arange(1, sx.size + 1)
            running_rms_x = np.sqrt(running_mean_x)
            self.rmsx_line.set_data(range(len(running_rms_x)), running_rms_x)
            self.rmsx_ax.set_xlim(0, max(10, len(running_rms_x)))
            self.rmsx_ax.set_ylim(0, max(0.1, running_rms_x.max() * 1.2))
        
        sy = np.array([v for v in self.sqerr_y[:current_idx] if not (v is None or np.isnan(v))], dtype=float)
        if sy.size > 0:
            running_mean_y = np.cumsum(sy) / np.arange(1, sy.size + 1)
            running_rms_y = np.sqrt(running_mean_y)
            self.rmsy_line.set_data(range(len(running_rms_y)), running_rms_y)
            self.rmsy_ax.set_xlim(0, max(10, len(running_rms_y)))
            self.rmsy_ax.set_ylim(0, max(0.1, running_rms_y.max() * 1.2))
        
        # --- Cálculo dos Erros Assinados (dx, dy) para o frame atual ---
        signed_dx = []
        signed_dy = []
        for m_pt, f_pt in zip(self.meas_pts[:current_idx], self.filt_pts[:current_idx]):
            if m_pt is not None and f_pt is not None:
                signed_dx.append(f_pt[0] - m_pt[0])
                signed_dy.append(f_pt[1] - m_pt[1])

        # --- Atualização do Histograma ---
        self.hist_ax.clear()
        self._style_ax(self.hist_ax)
        if signed_dx and signed_dy:
            self.hist_ax.hist(signed_dx, bins=20, alpha=0.5, color='blue', label='Erros X')
            self.hist_ax.hist(signed_dy, bins=20, alpha=0.5, color='orange', label='Erros Y')
            self.hist_ax.legend(loc='upper right', fontsize=7, facecolor="#f5f5f5", edgecolor="#999999")
        
        # --- Atualização da Dispersão ---
        self.scatter_ax.clear()
        self._style_ax(self.scatter_ax)
        if signed_dx and signed_dy:
            self.scatter_ax.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=15)
            self.scatter_ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            self.scatter_ax.axvline(0, color='black', linewidth=1, alpha=0.5)
            
            # Ajusta os limites para manter a origem (0,0) centralizada e acompanhar erros maiores
            max_err = max(max(np.abs(signed_dx)), max(np.abs(signed_dy)), 0.1)
            self.scatter_ax.set_xlim(-max_err * 1.2, max_err * 1.2)
            self.scatter_ax.set_ylim(-max_err * 1.2, max_err * 1.2)

        # Redesenha os canvas para a interface gráfica exibir
        self.rmsx_canvas.draw_idle()
        self.rmsy_canvas.draw_idle()
        self.hist_canvas.draw_idle()
        self.scatter_canvas.draw_idle()

def run_app():
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()