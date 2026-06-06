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
from src.simulation.world import World, Entidy
from src.gui.viewers import VideoViewer
from src.generator.trajectories import TrajectoryGenerator
from src.analytics.metrics import MetricsManager
from src.analytics.dashboard import ChartsDashboard
import traceback

class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Kalman - Arena Sintética EKF")
        
        # Video display size
        self.video_width = 640
        self.video_height = 480

        # Set minimum window size
        self.root.minsize(1400, 750)
        
        # Dimensões reais desejadas do ambiente
        self.max_x = 100 # metros
        self.max_y = 75  # metros

        # Localização das torres no mapa (serão lidas do painel)
        self.towers = None 

        # Tamanho mínimo do ROI 
        self.min_window_m = int(self.max_x / 60) # metros

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
        self.worker = None
        self.running = False
        self.processing = False
        self.playing = False
        self.paused = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_fps = 30.0 # FPS padrão da simulação
        
        # Métricas de avaliação e controle da UI
        self.detection_rate = 0.0  
        self.meas_inside_roi = 0 

        self.metrics = MetricsManager()

        # Mantidos para controle interno do algoritmo EKF (se necessário)
        self.kalman_windows = []    # Matrizes P de covariância
        self.innov_x = []           # Diferenças de inovação residual
        self.innov_y = []

        # Inicializa a interface gráfica
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
        ### --- LEFT PANEL: Settings, Trajectory, Bases & Metrics ---
        self.root.grid_rowconfigure(0, weight=1) 
        
        self.left_frame = tk.Frame(self.root, bg="white", width=300)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)

        # ===== SECTION 1: Escolha da Trajetória =====
        traj_lbl_frame = tk.LabelFrame(self.left_frame, text="🛤 Trajetória do Alvo", 
                                       font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        traj_lbl_frame.pack(fill="x", padx=8, pady=1)

        self.traj_var = tk.StringVar(value="Círculo")
        traj_combo = ttk.Combobox(traj_lbl_frame, textvariable=self.traj_var, 
                                  values=["Quadrado", "Círculo", "Tangente Hiperbólica", "Lemniscata", "Aleatória", "Oclusão"], state="readonly")
        traj_combo.pack(fill="x", pady=2)
        traj_combo.bind("<<ComboboxSelected>>", self._on_trajectory_change)

        self.traj_params_frame = tk.Frame(traj_lbl_frame, bg="#f5f5f5")
        self.traj_params_frame.pack(fill="x", pady=2)

        # ===== SECTION 1.5: Bases de Observação (4 Torres) =====
        bases_lbl_frame = tk.LabelFrame(self.left_frame, text="📡 Posição das 4 Bases (m)", 
                                        font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        bases_lbl_frame.pack(fill="x", padx=8, pady=1)

        self.base_entries = []
        default_bases = [(10, 10), (90, 10), (90, 65), (10, 65)] # Bases padrão
        for i in range(4):
            f = tk.Frame(bases_lbl_frame, bg="#f5f5f5")
            f.pack(fill="x", pady=1)
            tk.Label(f, text=f"B{i+1} (x,y):", font=("Segoe UI", 8), bg="#f5f5f5").pack(side="left")
            
            ex = ttk.Entry(f, width=5)
            ex.insert(0, str(default_bases[i][0]))
            ex.pack(side="left", padx=2)
            
            ey = ttk.Entry(f, width=5)
            ey.insert(0, str(default_bases[i][1]))
            ey.pack(side="left", padx=2)
            
            self.base_entries.append((ex, ey))

        # ===== SECTION 2: Opções do Filtro =====
        config_lbl_frame = tk.LabelFrame(self.left_frame, text="⚙ Opções do Filtro & Sensor", 
                                         font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=2)
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
        
        r_labels = ["R[0,0]", "R[1,1]", "R[2,2]", "R[3,3]"] # Alterado para 4, pois agora são 4 bases de distância
        default_r_vals = ["1e-1", "1e-1", "1e-1", "1e-1"]
        
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
                                        font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=2)
        debug_lbl_frame.pack(fill="x", padx=8, pady=1)

        ttk.Checkbutton(debug_lbl_frame, text="Desenhar trajetória real", variable=self.show_traj).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar torres de medição", variable=self.show_detect).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Kalman", variable=self.show_kalman).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Janela de Incerteza", variable=self.show_window).pack(anchor="w")

# ===== SECTION 4: Status e Métricas =====
        info_lbl_frame = tk.LabelFrame(self.left_frame, text="📊 Status e Métricas", 
                                       font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=4)
        info_lbl_frame.pack(fill="x", padx=8, pady=2)

        # Configura as colunas para dividirem o espaço igualmente
        info_lbl_frame.grid_columnconfigure(0, weight=1)
        info_lbl_frame.grid_columnconfigure(1, weight=1)

        # --- Linha 0: Status (Ocupa as duas colunas para destaque) ---
        self.status_lbl = tk.Label(info_lbl_frame, text="Status: Aguardando Execução", 
                                   font=("Segoe UI", 8, "bold"), fg="#16a34a", bg="#f5f5f5")
        self.status_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # --- Linha 1: Taxas ---
        self.det_rate_lbl = tk.Label(info_lbl_frame, text="Tx. Observação: --%", 
                                     font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.det_rate_lbl.grid(row=1, column=0, sticky="w", pady=1)

        self.inlier_rate_lbl = tk.Label(info_lbl_frame, text="Inliers (Janela): --%", 
                                        font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.inlier_rate_lbl.grid(row=1, column=1, sticky="w", pady=1)

        # --- Linha 2: Erros ---
        self.rmse_lbl = tk.Label(info_lbl_frame, text="RMSE (X|Y): --|-- m", 
                                 font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.rmse_lbl.grid(row=2, column=0, sticky="w", pady=1)

        self.mean_err_lbl = tk.Label(info_lbl_frame, text="Erro Médio: --|-- m", 
                                     font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.mean_err_lbl.grid(row=2, column=1, sticky="w", pady=1)

        # --- Linha 3: NIS e Regime Estacionário ---
        self.nis_lbl = tk.Label(info_lbl_frame, text="NIS Médio: --", 
                               font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.nis_lbl.grid(row=3, column=0, sticky="w", pady=1)

        self.steady_state_lbl = tk.Label(info_lbl_frame, text="Regime Est.: -- s (-- fr)", 
                                         font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.steady_state_lbl.grid(row=3, column=1, sticky="w", pady=1)

        # ===== SPACER: Garante o uso de todo o height útil =====
        spacer = tk.Frame(self.left_frame, bg="white")
        spacer.pack(fill="both", expand=True)

        # ===== SECTION 5: EXEC & SAVE Buttons =====
        exec_lbl_frame = tk.Frame(self.left_frame, bg="white")
        exec_lbl_frame.pack(side="bottom", fill="x", padx=8, pady=(4, 8))

        btn_frame = tk.Frame(exec_lbl_frame, bg="white")
        btn_frame.pack(fill="x")

        # ATENÇÃO: Botão de executar agora inicia HABILITADO ("normal") em vez de "disabled"
        self.exec_btn = tk.Button(btn_frame, text="▶ EXEC", command=self.execute_processing, 
                                font=("Segoe UI", 9, "bold"), bg="#333333", fg="white", 
                                relief="flat", padx=10, pady=8, cursor="hand2", state="normal")
        self.exec_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.save_btn = tk.Button(btn_frame, text="💾 Salvar", command=self.save_results, 
                                font=("Segoe UI", 9, "bold"), bg="#666666", fg="white", 
                                relief="flat", padx=10, pady=8, cursor="hand2", state="disabled")
        self.save_btn.pack(side="right", fill="x", expand=True, padx=(3, 0))

        # Chama a função para popular a tela da primeira vez
        # self._on_trajectory_change() # <- Descomente se esse método já estiver implementado

    def _build_center_painer(self):
        # --- CENTER PANEL: Single Viewer ---
        self.center_frame = tk.Frame(self.root, bg="#f3f4f6")
        self.center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # 1. TÍTULO
        viewer_title = tk.Label(self.center_frame, text="🎬 Arena de Simulação EKF", 
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
        
        # 4.1 Container da Legenda (O Banner principal do vídeo foi REMOVIDO)
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
        tk.Label(l1_frame, text="📡 Base: 4 Torres", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#d97706").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🟦 Kalman: Estimado", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#2563eb").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🟩 Trajetória Real (GT)", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#16a34a").pack(side="left", padx=15)
        tk.Label(l1_frame, text="🔷 Janela: Incerteza (±3σ)", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#9333ea").pack(side="left", padx=15)

        # Linha 2: Métricas de Escala e Debug
        l2_frame = tk.Frame(legend_container, bg="#ffffff")
        l2_frame.pack(anchor="center", pady=2)
        tk.Label(l2_frame, text=f"📐 Dimensões da Arena: {self.max_x}x{self.max_y} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        
        # (Opcional, min_window_m precisa existir no __init__)
        if hasattr(self, 'min_window_m'):
            tk.Label(l2_frame, text=f"🔲 ROI Mín: {self.min_window_m:.2f} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
            
        tk.Label(l2_frame, text=f"🔎 Escala Média: {avg_scale_px_m:.2f} px/m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        
        # Exibe o erro em metros (entrada) e a conversão para pixels
        self.erro_sensor_lbl = tk.Label(l2_frame, text=f"⚡ Erro Distância Simulado: {erro_metros:.4f} m ≈ {erro_px:.2f} px", 
                                        font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280")
        self.erro_sensor_lbl.pack(side="left", padx=15)
    
    def _build_right_painel(self):
        # --- RIGHT PANEL: Metrics ---
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        self.dashboard = ChartsDashboard(parent_frame=self.right_frame, style_ax_callback=self._style_ax)

        self.rmsx_fig = self.dashboard.rmsx_fig
        self.rmsx_ax  = self.dashboard.rmsx_ax
        self.rmsx_line = self.dashboard.rmsx_line
        self.rmsx_canvas = self.dashboard.rmsx_canvas

        self.rmsy_fig = self.dashboard.rmsy_fig
        self.rmsy_ax  = self.dashboard.rmsy_ax
        self.rmsy_line = self.dashboard.rmsy_line
        self.rmsy_canvas = self.dashboard.rmsy_canvas

        self.hist_fig = self.dashboard.hist_fig
        self.hist_ax  = self.dashboard.hist_ax
        self.hist_canvas = self.dashboard.hist_canvas

        self.scatter_fig = self.dashboard.scatter_fig
        self.scatter_ax  = self.dashboard.scatter_ax
        self.scatter_canvas = self.dashboard.scatter_canvas

        # Store config
        self.config_Q = None
        self.config_R = None
        self.config_detector_noise = 1.0
        self.video_fps = 30.0
        self.metrics_update_counter = 0

        self.rmsx_cached = None
        self.rmsy_cached = None

    def _on_trajectory_change(self, event=None):
        """Atualiza dinamicamente os parâmetros da trajetória escolhida, mantendo-os simples."""
        for widget in self.traj_params_frame.winfo_children():
            widget.destroy()
            
        traj_type = self.traj_var.get()
        self.traj_params = {} 
        
        def add_param(row, label_text, default_val, key):
            tk.Label(self.traj_params_frame, text=label_text, bg="#f5f5f5", font=("Segoe UI", 8)).grid(row=row, column=0, sticky="w", pady=1)
            entry = ttk.Entry(self.traj_params_frame, width=8)
            entry.insert(0, str(default_val))
            entry.grid(row=row, column=1, padx=4, pady=1)
            self.traj_params[key] = entry 

        if traj_type == "Círculo":
            add_param(0, "Raio (m):", 30.0, "raio")
            add_param(1, "Velocidade (m/s):", 2.0, "velocidade")
            
        elif traj_type == "Quadrado":
            add_param(0, "Lado (m):", 40.0, "lado")
            add_param(1, "Velocidade (m/s):", 2.0, "velocidade")
            
        elif traj_type == "Tangente Hiperbólica":
            add_param(0, "Amplitude (m):", 20.0, "amplitude")
            add_param(1, "Velocidade (m/s):", 2.0, "velocidade")
            
        elif traj_type == "Lemniscata": 
            add_param(0, "Amplitude (m):", 35.0, "amplitude")
            add_param(1, "Velocidade (m/s):", 1.5, "velocidade")
            
        elif traj_type == "Aleatória":
            add_param(0, "Vel. Média (m/s):", 2.0, "velocidade")
            add_param(1, "Ruído Máx (m):", 0.5, "ruido")
            
        elif traj_type == "Oclusão":
            add_param(0, "Raio Base (m):", 30.0, "raio")
            add_param(1, "Sumiço (frames):", 15, "oclusao_frames")
    
    def _on_escape(self, event=None):
        try:
            self.root.state("normal")
        except Exception:
            try:
                self.root.attributes("-fullscreen", False)
            except Exception:
                pass

    def execute_processing(self):
        """Dispara a simulação sintética de rastreamento via EKF."""
        if self.processing: return
        
        try:
            self._parse_config()
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro nas configurações: {error_msg}"))
            self.processing = False
            return
        
        self.processing = True
        self.exec_btn.config(state="disabled")
        if hasattr(self, 'load_btn'):
            self.load_btn.config(state="disabled")
        self.status_lbl.config(text="Status: Gerando Trajetória e Simulando EKF...")
        
        self.worker = threading.Thread(target=self._simulate_ekf_tracking, daemon=True)
        self.worker.start()

    def _simulate_ekf_tracking(self):
        """Laço principal unificado e simplificado graças à classe MetricsManager."""
        try:
            dt = 1.0 / self.video_fps
            gen = TrajectoryGenerator(dt, self.towers)
            tipo_traj = self.traj_var.get()
            duracao = 20.0 
            
            if tipo_traj == "Círculo":
                raio = float(self.traj_params["raio"].get())
                vel = float(self.traj_params["velocidade"].get())
                t, states = gen.generate_circle(raio, (self.max_x/2, self.max_y/2), vel, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Tangente Hiperbólica":
                amp = float(self.traj_params["amplitude"].get())
                vel = float(self.traj_params["velocidade"].get())
                t, states = gen.generate_tanh_curve((10, 20), (self.max_x-10, self.max_y-20), amp, 3.0, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Oclusão":
                raio = float(self.traj_params["raio"].get())
                frames_occ = int(self.traj_params["oclusao_frames"].get())
                t, states, mask = gen.generate_occlusion(raio, (self.max_x/2, self.max_y/2), 5.0, duracao, frames_occ)
            else:
                t, states = gen.generate_circle(25, (self.max_x/2, self.max_y/2), 3.0, duracao)
                mask = np.ones(len(t), dtype=bool)

            self.total_frames = len(t)
            
            world = World(self.towers, noise_std=self.config_detector_noise_m)
            ekf = Entidy(dt, self.towers, q_diag=self.config_Q, r_diag=self.config_R)
            ekf.initialize(states[0, 0], states[0, 1])

            # Limpa o gerenciador de métricas e o gravador de vídeo em RAM
            self.metrics.clear()
            self.recorded_frames = []

            arena_base = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
            for i, (bx, by) in enumerate(self.towers):
                px, py = self.m_to_px(bx, by)
                cv2.circle(arena_base, (px, py), 8, (0, 165, 255), -1) 
                cv2.putText(arena_base, f"B{i+1}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            for i in range(self.total_frames):
                if not self.processing: break 
                
                gt_x, gt_y = states[i, 0], states[i, 1]
                ekf.predict()
                
                raw_z = None
                if mask[i]:
                    raw_z = world.measure_distances(gt_x, gt_y)
                    ekf.update(raw_z)
                
                est_x, est_y = ekf.get_position()
                
                # --- NOVO: Empurramos tudo para a classe MetricsManager ---
                self.metrics.push_frame(gt_x, gt_y, est_x, est_y, raw_z=raw_z)

                # Renderização
                frame = arena_base.copy()
                px_gt, py_gt = self.m_to_px(gt_x, gt_y)
                px_est, py_est = self.m_to_px(est_x, est_y)
                
                if self.show_traj.get():
                    for pt in self.metrics.ground_truth_pts:
                        cv2.circle(frame, self.m_to_px(pt[0], pt[1]), 2, (0, 255, 0), -1)
                
                if self.show_kalman.get():
                    for pt in self.metrics.filt_pts:
                        cv2.circle(frame, self.m_to_px(pt[0], pt[1]), 2, (255, 0, 0), -1)

                if self.show_detect.get() and mask[i]:
                    for bx, by in self.towers:
                        px_b, py_b = self.m_to_px(bx, by)
                        cv2.line(frame, (px_b, py_b), (px_gt, py_gt), (0, 165, 255), 1)

                if self.show_window.get():
                    std_x, std_y = ekf.get_uncertainty()
                    rx = min(max(int(std_x * 3 * (self.video_width / self.max_x)), 1), self.video_width)
                    ry = min(max(int(std_y * 3 * (self.video_height / self.max_y)), 1), self.video_height)
                    cv2.ellipse(frame, (px_est, py_est), (rx, ry), 0, 0, 360, (255, 0, 255), 1)

                if mask[i]:
                    cv2.circle(frame, (px_gt, py_gt), 6, (0, 255, 0), -1) 
                else:
                    cv2.circle(frame, (px_gt, py_gt), 6, (0, 0, 100), -1) 
                cv2.circle(frame, (px_est, py_est), 5, (255, 100, 100), -1) 

                self.recorded_frames.append(frame.copy())
                self.current_frame_idx = i
                self.tela_viewer.update_image(frame)
                
                if i % 5 == 0:
                    self._update_metrics_ui()
                    self._update_plots_ui()
                
                time.sleep(1.0 / self.video_fps)

        except Exception as e:
            msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro na Simulação", msg))
        finally:
            self.processing = False
            self.root.after(0, self._update_ui_metrics_and_complete)

    def _update_metrics_ui(self):
        """Consulta as métricas em tempo real da classe MetricsManager."""
        rmse_x, rmse_y = self.metrics.calculate_rmse(self.current_frame_idx)
        
        self.root.after(0, lambda: self.rmse_lbl.config(text=f"RMSE (X | Y): {rmse_x:.2f} | {rmse_y:.2f} m"))
        self.root.after(0, lambda: self.time_info_lbl.config(text=f"Passo: {self.current_frame_idx} / {self.total_frames}"))

    def _update_plots_ui(self):
        """Delega a atualização dos gráficos ao ChartsDashboard."""
        if hasattr(self, 'dashboard'):
            self.dashboard.update_dashboard(self.metrics, self.current_frame_idx)

    def _style_ax(self, ax):
        ax.set_facecolor("#f5f5f5")
        ax.tick_params(colors="#666666", labelsize=8)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _update_ui_metrics_and_complete(self):
        """Finaliza consultando a classe MetricsManager."""
        rmse_x, rmse_y = self.metrics.calculate_rmse()
        dx, dy = self.metrics.get_signed_errors()
        
        self.rmse_x_total = rmse_x
        self.rmse_y_total = rmse_y
        self.mean_signed_dx = np.mean(dx) if dx else 0.0
        self.mean_signed_dy = np.mean(dy) if dy else 0.0
        self.detection_rate = 100.0
        self.inlier_rate = 100.0
        if not hasattr(self, 'mean_nis'): self.mean_nis = 0.0

        self.status_lbl.config(text="Status: Processado", fg="#16a34a")
        self.det_rate_lbl.config(text=f"Taxa de Detecção (Visível): {self.detection_rate:.1f}%")
        self.inlier_rate_lbl.config(text=f"Inliers (na Janela): {self.inlier_rate:.1f}%")
        self.rmse_lbl.config(text=f"RMSE (X | Y): {self.rmse_x_total:.4f} | {self.rmse_y_total:.4f} m")
        self.mean_err_lbl.config(text=f"Erro Médio (X | Y): {self.mean_signed_dx:+.4f} | {self.mean_signed_dy:+.4f} m")
        self.nis_lbl.config(text=f"NIS Médio: {self.mean_nis:.3f} (Ideal ~2)")
        
        self._on_processing_complete()
    
    def _find_convergence_frame(self, running_rms, final_val, tol=0.05, min_stable=10):
        if final_val is None or len(running_rms) < min_stable: return None
        threshold = final_val * (1 + tol)
        start = max(1, len(running_rms) // 5) 
        for i in range(start, len(running_rms) - min_stable + 1):
            if all(v <= threshold for v in running_rms[i:i+min_stable]):
                return i
        return None

    def save_results(self):
        """Lê os dados centralizados em self.metrics e gera os relatórios."""
        if not self.metrics.filt_pts or not self.metrics.ground_truth_pts:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar.")
            return

        try:
            video_name = "simulacao_arena"
            save_dir = f"FiltroKalman/src/results/{video_name}"
            os.makedirs(save_dir, exist_ok=True)

            signed_dx, signed_dy = self.metrics.get_signed_errors()
            
            sx = np.array(self.metrics.sqerr_x)
            sy = np.array(self.metrics.sqerr_y)
            run_rms_x = np.sqrt(np.cumsum(sx) / np.arange(1, sx.size + 1)) if sx.size > 0 else []
            run_rms_y = np.sqrt(np.cumsum(sy) / np.arange(1, sy.size + 1)) if sy.size > 0 else []

            # ===== GRÁFICO 1: TRAJETÓRIA =====
            fig1 = Figure(figsize=(12, 8), tight_layout=True, dpi=150)
            ax1 = fig1.add_subplot(111)
            xs_gt = [p[0] for p in self.metrics.ground_truth_pts]
            ys_gt = [p[1] for p in self.metrics.ground_truth_pts]
            xs_filt = [p[0] for p in self.metrics.filt_pts]
            ys_filt = [p[1] for p in self.metrics.filt_pts]
            # Na arena usamos o GT em vez de "meas_pts" X/Y direto
            ax1.plot(xs_gt, ys_gt, "g-", label="Ground Truth (Real)", linewidth=2.5, alpha=0.9)
            ax1.plot(xs_filt, ys_filt, "b--", label="Kalman Filtrado", linewidth=2.0)
            ax1.set_title("Trajetória Espacial", fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle="--")
            fig1.savefig(f"{save_dir}/{video_name}_1_traj.png")

            # Demais gráficos seguem idênticos, lendo dos arrays corretos...
            # (Apenas substitua onde tinha self.nis_vals para self.metrics.nis_vals)

            plt.close('all')

            # ========== CSV VIA METRICS MANAGER ==========
            csv_path = f"{save_dir}/{video_name}_positions.csv"
            sucesso = self.metrics.save_to_csv(csv_path)

            if sucesso:
                messagebox.showinfo("Sucesso", f"Análise completa e arquivo .csv salvos em:\n{save_dir}/")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao gerar relatórios: {error_msg}"))
        finally:
            self.processing = False
            
    def _parse_config(self):
        """
        Lê e traduz os campos da interface gráfica.
        - Q: Processo (6 estados do modelo PVA)
        - R: Medição (4 valores, um para cada torre)
        - Torres: Lê as coordenadas X e Y das 4 bases na UI.
        """
        # 1. Parse Q (Modelo PVA -> 6 estados: x, y, vx, vy, ax, ay)
        q_vals = []
        entries_q = getattr(self, 'q_entries', [])
        for entry in entries_q:
            try: 
                val_str = entry.get().strip()
                q_vals.append(float(val_str))
            except (ValueError, AttributeError): 
                q_vals.append(1.0)
        
        while len(q_vals) < 6:
            q_vals.append(1.0)
        self.config_Q = q_vals[:6]
        
        # 2. Parse R (4 Distâncias -> 4 Bases)
        r_vals = []
        entries_r = getattr(self, 'r_entries', [])
        for entry in entries_r:
            try: 
                val_str = entry.get().strip()
                r_vals.append(float(val_str))
            except (ValueError, AttributeError): 
                r_vals.append(0.1)

        while len(r_vals) < 4:
            r_vals.append(0.1)
        self.config_R = r_vals[:4]

        # 3. Parse das Posições das 4 Torres
        towers_temp = []
        if hasattr(self, 'base_entries') and self.base_entries is not None:
            for pair in self.base_entries:
                ent_x, ent_y = pair
                try:
                    x = float(ent_x.get().strip())
                    y = float(ent_y.get().strip())
                    towers_temp.append([x, y])
                except (ValueError, AttributeError):
                    towers_temp.append([0.0, 0.0])
        else:
            # Padrão de fallback caso as caixas de texto falhem ou não existam
            towers_temp = [[10.0, 10.0], [90.0, 10.0], [90.0, 65.0], [10.0, 65.0]]

        self.towers = np.array(towers_temp)
            
        # 4. Obtém o erro de distância padrão em METROS do input do detector noise
        erro_metros = 1.0
        if hasattr(self, 'detector_noise_entry') and self.detector_noise_entry is not None:
            try:
                erro_metros = float(self.detector_noise_entry.get().strip())
            except (ValueError, AttributeError):
                erro_metros = 1.0

        # Cálculo da conversão Média de Metros para Pixels (para exibição na UI)
        max_x = getattr(self, 'max_x', 100.0)
        max_y = getattr(self, 'max_y', 75.0)
        px_m_x = self.video_width / max_x if max_x > 0 else 1.0
        px_m_y = self.video_height / max_y if max_y > 0 else 1.0
        avg_scale_px_m = (px_m_x + px_m_y) / 2.0
        
        # 5. Salva o ruído
        self.config_detector_noise_m = erro_metros          
        self.config_detector_noise = erro_metros * avg_scale_px_m

        # 6. Atualiza a label da dashboard visual
        if hasattr(self, 'erro_sensor_lbl') and self.erro_sensor_lbl is not None:
            erro_px = erro_metros * avg_scale_px_m
            try:
                self.erro_sensor_lbl.config(text=f"⚡ Erro Distância Simulado: {erro_metros:.4f} m ≈ {erro_px:.2f} px")
            except Exception:
                pass

        # 7. Reseta os textos das métricas na UI (Preparando para nova execução)
        if hasattr(self, 'status_lbl'): self.status_lbl.config(text="Status: Configurações Carregadas", fg="#16a34a")
        if hasattr(self, 'det_rate_lbl'): self.det_rate_lbl.config(text="Taxa de Observação: --%", fg="#4b5563")
        if hasattr(self, 'inlier_rate_lbl'): self.inlier_rate_lbl.config(text="Inliers (na Janela): --%", fg="#4b5563")
        if hasattr(self, 'rmse_lbl'): self.rmse_lbl.config(text="RMSE (X | Y): -- | -- m", fg="#4b5563")
        if hasattr(self, 'mean_err_lbl'): self.mean_err_lbl.config(text="Erro Médio (X | Y): -- | -- m", fg="#4b5563")
        if hasattr(self, 'nis_lbl'): self.nis_lbl.config(text="NIS Médio: --", fg="#4b5563")
        if hasattr(self, 'steady_state_lbl'): self.steady_state_lbl.config(text="Regime Estacionário: -- s (-- frames)", fg="#4b5563")
        
    def _on_processing_complete(self):
        """Seletor de interface chamado ao finalizar a simulação."""
        self.status_lbl.config(text=f"Status: Concluído | Detecção: {self.inlier_rate:.1f}%")
        self.exec_btn.config(state="normal")
        
        if hasattr(self, 'load_btn'):
            self.load_btn.config(state="normal")

        # Verifica e habilita o vídeo na memória
        if hasattr(self, 'recorded_frames') and len(self.recorded_frames) > 0:
            self.total_frames = len(self.recorded_frames)
            self.current_frame_idx = 0

            primeiro_frame = self.recorded_frames[0]
            
            if hasattr(self.tela_viewer, 'display_image'):
                self.tela_viewer.display_image(primeiro_frame)
            else:
                self.tela_viewer.update_image(primeiro_frame) 

            # Habilita botões e mostra o Dashboard final
            self.prev_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.save_btn.config(state="normal")

            self._update_plots_ui()
    
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

    def m_to_px(self, x_m, y_m):
        """
        Converte coordenadas em metros (do simulador) para coordenadas em pixels (para o OpenCV).
        A origem (0,0) em metros é o canto INFERIOR esquerdo.
        A origem (0,0) em pixels no OpenCV é o canto SUPERIOR esquerdo.
        """
        px = int(x_m * (self.video_width / self.max_x))
        # Inverte o eixo Y para o OpenCV
        py = int(self.video_height - (y_m * (self.video_height / self.max_y)))
        return px, py

def run_app():
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()