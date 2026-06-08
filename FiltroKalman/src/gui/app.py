import threading
import time
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
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
        


        # Set minimum window size
        self.root.minsize(1400, 750)
        
        ## Esses valores são ajustados no build-center_painel, então aqui são apenas os valores default!
        # Video display size default
        self.video_width = 640
        self.video_height = 480
        # Dimensões reais default
        self.max_x = 150 # metros
        self.max_y = 150  # metros

        # Localização das torres no mapa (serão lidas do painel)
        self.towers = None 

        # Tamanho mínimo do ROI 
        self.min_window_m = int(self.max_x / 80) # metros

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
        self.video_fps = 60.0 # FPS padrão da simulação
        
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
        self.show_kalman_traj = tk.BooleanVar(value=True)

        # CORREÇÃO: Alinhando os nomes das variáveis para corresponderem ao loop e ao trace
        self.show_traj_cache = True
        self.show_detect_cache = True
        self.show_kalman_cache = True
        self.show_window_cache = True
        
        # Configura traces para manter os caches atualizados
        self.show_traj.trace_add('write', lambda *_: setattr(self, 'show_traj_cache', self.show_traj.get()))
        self.show_detect.trace_add('write', lambda *_: setattr(self, 'show_detect_cache', self.show_detect.get()))
        self.show_kalman.trace_add('write', lambda *_: setattr(self, 'show_kalman_cache', self.show_kalman.get()))
        self.show_window.trace_add('write', lambda *_: setattr(self, 'show_window_cache', self.show_window.get()))
        
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

        self.left_frame.pack_propagate(False)

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
        w_def = 180
        h_def = 120
        c_def = 10

        default_bases = [(c_def, c_def), (c_def+w_def, c_def), (c_def+w_def, c_def+h_def), (c_def, c_def+h_def)] # Bases padrão
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
        default_q_vals = ["3", "3", "3", "3", "3", "3"]
        
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
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Kalman (Atual)", variable=self.show_kalman).pack(anchor="w")
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar trajetória de Kalman", variable=self.show_kalman_traj).pack(anchor="w") # <-- ADICIONADO AQUI
        ttk.Checkbutton(debug_lbl_frame, text="Desenhar Janela de Incerteza", variable=self.show_window).pack(anchor="w")

        # ===== SECTION 4: Status e Métricas =====
        info_lbl_frame = tk.LabelFrame(self.left_frame, text="📊 Status e Métricas", 
                                       font=("Segoe UI", 9, "bold"), bg="#f5f5f5", fg="#333333", padx=8, pady=4)
        info_lbl_frame.pack(fill="x", padx=8, pady=2)

        # Configura as colunas para dividirem o espaço igualmente
        info_lbl_frame.grid_columnconfigure(0, weight=1)
        info_lbl_frame.grid_columnconfigure(1, weight=1)

        # --- Linha 0: Status (Ocupa as duas colunas para destaque) ---
        self.status_lbl = tk.Label(info_lbl_frame, text="Status: Aguardando", 
                                   font=("Segoe UI", 8, "bold"), fg="#16a34a", bg="#f5f5f5")
        self.status_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # --- Linha 1: Taxas ---
        self.det_rate_lbl = tk.Label(info_lbl_frame, text="Tx. Detec: --%", 
                                     font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.det_rate_lbl.grid(row=1, column=0, sticky="w", pady=1)

        self.inlier_rate_lbl = tk.Label(info_lbl_frame, text="Inliers : --%", 
                                        font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.inlier_rate_lbl.grid(row=1, column=1, sticky="w", pady=1)

        # --- Linha 2: Erros ---
        self.rmse_lbl = tk.Label(info_lbl_frame, text="RMSE (X|Y): --|-- m", 
                                 font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.rmse_lbl.grid(row=2, column=0, sticky="w", pady=1)

        self.mean_err_lbl = tk.Label(info_lbl_frame, text="E_med: --|-- m", 
                                     font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.mean_err_lbl.grid(row=2, column=1, sticky="w", pady=1)

        # --- Linha 3: NIS e Regime Estacionário ---
        self.nis_lbl = tk.Label(info_lbl_frame, text="NIS_m: --", 
                               font=("Segoe UI", 8), fg="#4b5563", bg="#f5f5f5")
        self.nis_lbl.grid(row=3, column=0, sticky="w", pady=1)

        self.steady_state_lbl = tk.Label(info_lbl_frame, text="Reg. Est.: -- s (-- fr)", 
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

        self._on_trajectory_change()

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

        self.root.update_idletasks()
        
        lw = viewer_container.winfo_width()
        lh = viewer_container.winfo_height()
        
        if lw > 1 and lh > 1:
            self.video_width = lw
            self.video_height = lh
            
        # Fixamos o Y em 150 metros e calculamos o X para que a escala px/m seja idêntica
        aspect_ratio = self.video_width / self.video_height
        self.max_y = 150.0
        self.max_x = self.max_y * aspect_ratio
        
        # Atualiza a janela mínima de ROI com base na nova escala
        self.min_window_m = int(self.max_x / 80)
        # =====================================================================
        
        # Agora inicializamos o VideoViewer com as dimensões corretas do monitor
        self.tela_viewer = VideoViewer(viewer_container, width=self.video_width, 
                                    height=self.video_height, bg="black")
        self.tela_viewer.pack(fill="both", expand=True)

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
        
        legend_container = tk.Frame(info_dashboard, bg="#ffffff")
        legend_container.pack(fill="x", pady=(6, 12))
        
        # Como ajustamos o aspect_ratio, px_m_x será exatmente igual ao px_m_y
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
        
        # O max_x agora exibirá um número quebrado exato (ex: 150x266.67 m), mostrando a nova largura simulada da arena
        tk.Label(l2_frame, text=f"📐 Dimensões da Arena: {self.max_x:.1f}x{self.max_y:.1f} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        
        if hasattr(self, 'min_window_m'):
            tk.Label(l2_frame, text=f"🔲 ROI Mín: {self.min_window_m:.2f} m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
            
        tk.Label(l2_frame, text=f"🔎 Escala Exata: {avg_scale_px_m:.2f} px/m", font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280").pack(side="left", padx=15)
        
        self.erro_sensor_lbl = tk.Label(l2_frame, text=f"⚡ Erro Distância Simulado: {erro_metros:.2f} m ≈ {erro_px:.2f} px", 
                                        font=("Segoe UI", 9), bg="#ffffff", fg="#6b7280")
        self.erro_sensor_lbl.pack(side="left", padx=15)

    def _build_right_painel(self):
        # --- RIGHT PANEL: Metrics ---
        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

        self.dashboard = ChartsDashboard(parent_frame=self.right_frame, style_ax_callback=self._style_ax)

        # 1. RMS (agora unificado)
        self.rms_fig = self.dashboard.rms_fig
        self.rms_ax  = self.dashboard.rms_ax
        self.rms_canvas = self.dashboard.rms_canvas

        # 2. Histograma
        self.hist_fig = self.dashboard.hist_fig
        self.hist_ax  = self.dashboard.hist_ax
        self.hist_canvas = self.dashboard.hist_canvas

        # 3. Scatter
        self.scatter_fig = self.dashboard.scatter_fig
        self.scatter_ax  = self.dashboard.scatter_ax
        self.scatter_canvas = self.dashboard.scatter_canvas

        # 4. NIS (NOVO)
        self.nis_fig = self.dashboard.nis_fig
        self.nis_ax  = self.dashboard.nis_ax
        self.nis_canvas = self.dashboard.nis_canvas

        # Store config
        self.config_Q = None
        self.config_R = None
        self.config_detector_noise = 1.0
        self.video_fps = 30.0
        self.metrics_update_counter = 0

        self.rms_cached = None

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
        """Simulação rápida. Salva os frames no HD e não atualiza gráficos durante o loop."""
        out = None
        try:
            # --- CAPTURA SEGURA DA NOVA OPÇÃO DE DEBUG PARA A THREAD ---
            show_kalman_traj_cache = self.show_kalman_traj.get() if hasattr(self, 'show_kalman_traj') else True

            dt = 1.0 / self.video_fps
            gen = TrajectoryGenerator(dt, self.towers)
            tipo_traj = self.traj_var.get()
            duracao = 20.0

            if tipo_traj == "Círculo":
                raio = float(self.traj_params["raio"].get())
                vel = float(self.traj_params["velocidade"].get())
                t, states = gen.generate_circle(raio, (self.max_x / 2, self.max_y / 2), vel, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Quadrado":
                lado = float(self.traj_params["lado"].get())
                vel = float(self.traj_params["velocidade"].get())
                bottom_left = (self.max_x / 2 - lado / 2, self.max_y / 2 - lado / 2)
                t, states = gen.generate_square(lado, bottom_left, vel)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Tangente Hiperbólica":
                amp = float(self.traj_params["amplitude"].get())
                vel = float(self.traj_params["velocidade"].get())
                t, states = gen.generate_tanh_curve((10, 20), (self.max_x - 10, self.max_y - 20), amp, 3.0, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Lemniscata":
                amp = float(self.traj_params["amplitude"].get())
                vel = float(self.traj_params["velocidade"].get())
                t, states = gen.generate_lemniscate(amp, (self.max_x / 2, self.max_y / 2), vel, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Aleatória":
                vel = float(self.traj_params["velocidade"].get())
                ruido = float(self.traj_params["ruido"].get())
                t, states = gen.generate_random((self.max_x / 2, self.max_y / 2), vel, ruido, duracao)
                mask = np.ones(len(t), dtype=bool)
            elif tipo_traj == "Oclusão":
                raio = float(self.traj_params["raio"].get())
                frames_occ = int(self.traj_params["oclusao_frames"].get())
                t, states, mask = gen.generate_occlusion(raio, (self.max_x / 2, self.max_y / 2), 5.0, duracao, frames_occ)
            else:
                t, states = gen.generate_circle(25, (self.max_x / 2, self.max_y / 2), 3.0, duracao)
                mask = np.ones(len(t), dtype=bool)

            self.total_frames = len(t)

            world = World(self.towers, noise_std=self.config_detector_noise_m)
            ekf = Entidy(dt, self.towers, q_diag=self.config_Q, r_diag=self.config_R)
            ekf.initialize(states[0, 0], states[0, 1])

            self.metrics.clear()

            os.makedirs("FiltroKalman/src/data", exist_ok=True)
            output_path = "FiltroKalman/src/data/simulacao_arena.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, self.video_fps, (self.video_width, self.video_height))

            arena_base = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
            for i, (bx, by) in enumerate(self.towers):
                px, py = self.m_to_px(bx, by)
                cv2.circle(arena_base, (px, py), 8, (0, 165, 255), -1)
                cv2.putText(arena_base, f"B{i+1}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # --- LOOP PRINCIPAL ---
            for i in range(self.total_frames):
                if not self.processing:
                    break

                gt_x, gt_y = states[i, 0], states[i, 1]
                ekf.predict()

                raw_z = None
                nis_atual = np.nan
                meas_x, meas_y = None, None

                if mask[i]:
                    raw_z = world.measure_distances(gt_x, gt_y)
                    ekf.update(raw_z)
                    meas_x, meas_y = world.multilaterate(raw_z)

                    # Cálculo seguro do NIS
                    if hasattr(ekf, 'S') and hasattr(ekf, 'y') and ekf.S is not None and ekf.y is not None:
                        try:
                            try:
                                invS = np.linalg.inv(ekf.S)
                            except np.linalg.LinAlgError:
                                invS = np.linalg.pinv(ekf.S)
                            
                            nis_matriz = ekf.y.T @ invS @ ekf.y
                            nis_atual = float(nis_matriz.item())
                        except Exception:
                            nis_atual = np.nan

                est_x, est_y = ekf.get_position()
                P_mat_atual = ekf.P if hasattr(ekf, 'P') else None
                
                self.metrics.push_frame(gt_x, gt_y, est_x, est_y, meas_x=meas_x, meas_y=meas_y, nis_val=nis_atual, raw_z=raw_z, P_mat=P_mat_atual)

                frame = arena_base.copy()

                # Desenha histórico da trajetória Real (Verde)
                if self.show_traj_cache:
                    for pt in self.metrics.ground_truth_pts:
                        cv2.circle(frame, self.m_to_px(pt[0], pt[1]), 2, (0, 255, 0), -1)

                # --- NOVO: DESENHA O HISTÓRICO DA TRAJETÓRIA DE KALMAN (Azul claro/fino) ---
                if show_kalman_traj_cache:
                    for pt in self.metrics.filt_pts:
                        if pt is not None:
                            cv2.circle(frame, self.m_to_px(pt[0], pt[1]), 2, (255, 0, 0), -1)

                px_gt, py_gt = self.m_to_px(gt_x, gt_y)

                if self.show_detect_cache and mask[i]:
                    for j, (bx, by) in enumerate(self.towers):
                        px_b, py_b = self.m_to_px(bx, by)
                        cv2.line(frame, (px_b, py_b), (px_gt, py_gt), (0, 165, 255), 1)
                        
                        mid_x = int((px_b + px_gt) / 2)
                        mid_y = int((py_b + py_gt) / 2) - 8
                        
                        dist_medida = raw_z[j, 0]
                        cv2.putText(frame, f"{dist_medida:.1f}m", (mid_x, mid_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

                if self.show_window_cache and P_mat_atual is not None:
                    try:
                        std_x = np.sqrt(max(1e-6, P_mat_atual[0, 0]))
                        std_y = np.sqrt(max(1e-6, P_mat_atual[1, 1]))
                        win_x_m = max(self.min_window_m, std_x * 3)
                        win_y_m = max(self.min_window_m, std_y * 3)
                        
                        px1_raw, py1_raw = self.m_to_px(est_x - win_x_m, est_y + win_y_m)
                        px2_raw, py2_raw = self.m_to_px(est_x + win_x_m, est_y - win_y_m)
                        
                        x1 = int(max(0, min(px1_raw, self.video_width)))
                        y1 = int(max(0, min(py1_raw, self.video_height)))
                        x2 = int(max(0, min(px2_raw, self.video_width)))
                        y2 = int(max(0, min(py2_raw, self.video_height)))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (147, 51, 234), 2) 
                    except:
                        pass

                if mask[i]:
                    cv2.circle(frame, (px_gt, py_gt), 6, (0, 255, 0), -1)
                else:
                    cv2.circle(frame, (px_gt, py_gt), 6, (0, 0, 100), -1)

                # Desenha a posição atual (Instantânea) do Kalman (Círculo Azul Maior)
                if self.show_kalman_cache:
                    cv2.circle(frame, self.m_to_px(est_x, est_y), 6, (255, 0, 0), -1)

                out.write(frame)

                if i % 20 == 0:
                    self.root.after(0, lambda idx=i: self.status_lbl.config(text=f"Processando frame {idx}/{self.total_frames}..."))

            self.processed_video_path = output_path

        except Exception as e:
            msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro na Simulação", msg))
        finally:
            self.processing = False
            if out is not None:
                out.release()
            self.root.after(0, self._update_ui_metrics_and_complete)
    
    def save_results(self):
        """Salva gráficos detalhados, relatório e dados brutos em CSV em src/results/"""
        if not hasattr(self, 'metrics') or not self.metrics.filt_pts or not self.metrics.ground_truth_pts:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar.")
            return

        try:
            # Nome baseado na trajetória escolhida
            traj_name = self.traj_var.get().replace(" ", "_").lower()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = f"simulacao_{traj_name}"
            save_dir = f"FiltroKalman/src/results/{video_name}"
            os.makedirs(save_dir, exist_ok=True)

            # ========== Preparação de dados via MetricsManager ==========
            signed_dx, signed_dy = self.metrics.get_signed_errors()
            valid_nis = [n for n in self.metrics.nis_vals if not np.isnan(n) and not np.isinf(n)]

            sx = np.array([v for v in self.metrics.sqerr_x if not np.isnan(v)])
            sy = np.array([v for v in self.metrics.sqerr_y if not np.isnan(v)])
            run_rms_x = np.sqrt(np.cumsum(sx) / np.arange(1, sx.size + 1)) if sx.size > 0 else []
            run_rms_y = np.sqrt(np.cumsum(sy) / np.arange(1, sy.size + 1)) if sy.size > 0 else []

            from matplotlib.figure import Figure

            # ===== GRÁFICO 1: TRAJETÓRIA ESPACIAL =====
            fig1 = Figure(figsize=(12, 8), tight_layout=True, dpi=150)
            ax1 = fig1.add_subplot(111)
            
            xs_gt = [p[0] for p in self.metrics.ground_truth_pts]
            ys_gt = [p[1] for p in self.metrics.ground_truth_pts]
            xs_filt = [p[0] for p in self.metrics.filt_pts]
            ys_filt = [p[1] for p in self.metrics.filt_pts]

            # Pega as detecções da multilateração
            xs_det = [p[0] for p in self.metrics.meas_pts if p[0] is not None]
            ys_det = [p[1] for p in self.metrics.meas_pts if p[1] is not None]

            ax1.plot(xs_gt, ys_gt, "g:", label="Trajetória Real (GT)", linewidth=2.5, alpha=0.8)
            if xs_det and ys_det:
                ax1.plot(xs_det, ys_det, "r.", label="Pontos de Detecção (Com Ruído)", markersize=9, alpha=0.6)
            ax1.plot(xs_filt, ys_filt, "b-", label="Kalman Filtrado", linewidth=2.0, alpha=0.9)
            
            ax1.set_xlabel("Posição X (metros)", fontweight="bold")
            ax1.set_ylabel("Posição Y (metros)", fontweight="bold")
            ax1.set_title(f"Trajetória Espacial - {self.traj_var.get()}", fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle="--")
            fig1.savefig(f"{save_dir}/{video_name}_1_traj.png")

            # ===== GRÁFICO 2: EVOLUÇÃO DO ERRO RMS ACUMULADO =====
            fig2 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax2 = fig2.add_subplot(111)
            if len(run_rms_x) > 0:
                ax2.plot(run_rms_x, label="RMS X", color="blue", linewidth=1.5)
                ax2.plot(run_rms_y, label="RMS Y", color="orange", linewidth=1.5)
                
                final_x = run_rms_x[-1]
                final_y = run_rms_y[-1]
                
                # Procura frames de convergência usando a sua função interna
                if hasattr(self, '_find_convergence_frame'):
                    conv_idx_x = self._find_convergence_frame(run_rms_x, final_x)
                    conv_idx_y = self._find_convergence_frame(run_rms_y, final_y)
                    if conv_idx_x is not None:
                        ax2.axvline(conv_idx_x, color='blue', linestyle=':', alpha=0.7, label=f"Conv. X: {conv_idx_x/self.video_fps:.2f}s")
                    if conv_idx_y is not None:
                        ax2.axvline(conv_idx_y, color='orange', linestyle=':', alpha=0.7, label=f"Conv. Y: {conv_idx_y/self.video_fps:.2f}s")
                        
            ax2.set_xlabel("Frames", fontweight="bold")
            ax2.set_ylabel("Erro RMS Acumulado (metros)", fontweight="bold")
            ax2.set_title("Evolução do Erro RMS", fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3, linestyle="--")
            fig2.savefig(f"{save_dir}/{video_name}_2_rms.png")

            # ===== GRÁFICO 3: DISPERSÃO DOS ERROS =====
            fig3 = Figure(figsize=(8, 8), tight_layout=True, dpi=150)
            ax3 = fig3.add_subplot(111)
            ax3.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=20)
            ax3.axhline(0, color='black', linewidth=1, linestyle='-')
            ax3.axvline(0, color='black', linewidth=1, linestyle='-')
            ax3.set_xlabel("Erro X (m)", fontweight="bold")
            ax3.set_ylabel("Erro Y (m)", fontweight="bold")
            ax3.set_title("Dispersão dos Erros de Estimação (Assinados)", fontweight="bold")
            ax3.grid(True, alpha=0.3, linestyle="--")
            
            # Força eixos simétricos ao redor do zero para melhor leitura visual
            max_val = max(1.0, np.max(np.abs(signed_dx)) if signed_dx else 1.0, np.max(np.abs(signed_dy)) if signed_dy else 1.0)
            ax3.set_xlim(-max_val * 1.1, max_val * 1.1)
            ax3.set_ylim(-max_val * 1.1, max_val * 1.1)
            
            fig3.savefig(f"{save_dir}/{video_name}_3_scatter.png")

            # ===== GRÁFICO 4: TESTE DE CONSISTÊNCIA NIS =====
            fig4 = Figure(figsize=(12, 6), tight_layout=True, dpi=150)
            ax4 = fig4.add_subplot(111)
            
            # Filtra os índices válidos correspondentes para o eixo X
            frames_nis = [idx for idx, val in enumerate(self.metrics.nis_vals) if not np.isnan(val) and not np.isinf(val)]
            
            # Limite Chi-Quadrado dinâmico baseado na quantidade de torres (Ex: 4 torres = 4 graus de liberdade)
            num_towers = len(self.towers) if hasattr(self, 'towers') else 4
            if num_towers == 4:
                chi2_limit = 9.488   # 4 DOF @ 95%
                esp_nis = "4.0"
            elif num_towers == 3:
                chi2_limit = 7.815   # 3 DOF @ 95%
                esp_nis = "3.0"
            else:
                chi2_limit = 5.991   # 2 DOF @ 95% (padrão se for apenas posição X e Y puras)
                esp_nis = "2.0"
            
            if valid_nis:
                ax4.plot(frames_nis, valid_nis, color='#16a34a', alpha=0.8, linewidth=1.5, label="NIS Calculado")
                ax4.axhline(chi2_limit, color='r', linestyle='--', linewidth=2, label=f"Limite 95% Confiança (χ²={chi2_limit})")
                
                # Ajusta o teto visual para não deixar picos isolados esmagarem o gráfico
                teto_visual = float(np.percentile(valid_nis, 95) * 2.0)
                ax4.set_ylim(0, max(chi2_limit * 1.5, teto_visual))
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'Cálculo de NIS Indisponível no histórico', ha='center', va='center', transform=ax4.transAxes, color='red', fontsize=12)
                
            ax4.set_xlabel("Frames", fontweight="bold")
            ax4.set_ylabel("Valor NIS", fontweight="bold")
            ax4.set_title(f"Teste de Consistência NIS (Inovação Normalizada - Esperado ~{esp_nis})", fontweight="bold")
            ax4.grid(True, alpha=0.3, linestyle="--")
            fig4.savefig(f"{save_dir}/{video_name}_4_nis.png")

            # ===== GRÁFICO 5: HISTOGRAMA DOS ERROS =====
            fig5 = Figure(figsize=(10, 6), tight_layout=True, dpi=150)
            ax5 = fig5.add_subplot(111)
            ax5.hist(signed_dx, bins=30, alpha=0.5, color='blue', label='Erros X (m)', edgecolor='black', linewidth=0.5)
            ax5.hist(signed_dy, bins=30, alpha=0.5, color='orange', label='Erros Y (m)', edgecolor='black', linewidth=0.5)
            ax5.set_xlabel("Erro (metros)", fontweight="bold")
            ax5.set_ylabel("Frequência", fontweight="bold")
            ax5.set_title("Histograma dos Erros de Estado", fontweight="bold")
            ax5.legend()
            ax5.grid(True, alpha=0.3, linestyle="--")
            fig5.savefig(f"{save_dir}/{video_name}_5_hist.png")

            # ========== CÁLCULO DE MÉTRICAS COMPLEMENTARES ==========
            rmse_x, rmse_y = self.metrics.calculate_rmse()
            mean_dx = np.mean(signed_dx) if signed_dx else 0.0
            mean_dy = np.mean(signed_dy) if signed_dy else 0.0
            mean_nis_val = np.mean(valid_nis) if valid_nis else 0.0

            std_dx = np.std(signed_dx) if signed_dx else 0.0
            std_dy = np.std(signed_dy) if signed_dy else 0.0
            max_err_x = np.max(np.abs(signed_dx)) if signed_dx else 0.0
            max_err_y = np.max(np.abs(signed_dy)) if signed_dy else 0.0

            nis_above_95 = sum(1 for n in valid_nis if n > chi2_limit)
            nis_pct_above = (nis_above_95 / len(valid_nis)) * 100 if valid_nis else 0.0

            conv_time_x, conv_time_y = None, None
            if hasattr(self, '_find_convergence_frame'):
                conv_idx_x = self._find_convergence_frame(run_rms_x, run_rms_x[-1] if len(run_rms_x) else None)
                conv_idx_y = self._find_convergence_frame(run_rms_y, run_rms_y[-1] if len(run_rms_y) else None)
                conv_time_x = conv_idx_x / self.video_fps if conv_idx_x is not None else None
                conv_time_y = conv_idx_y / self.video_fps if conv_idx_y is not None else None

            # ========== RELATÓRIO TXT COMPLETO ==========
            txt_path = f"{save_dir}/{video_name}_metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("====================================================\n")
                f.write("       RESUMO DE MÉTRICAS - FILTRO DE KALMAN EKF     \n")
                f.write("====================================================\n\n")
                f.write(f"Cenário Sintético: {self.traj_var.get()}\n")
                f.write(f"Data da Análise: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Total de Frames: {self.total_frames}\n")
                f.write(f"FPS da Simulação: {self.video_fps:.2f}\n\n")

                f.write("--- PARÂMETROS DO FILTRO ---\n")
                f.write("Vetor de Variâncias de Processo (Q Diagonal Config):\n")
                f.write(f"{self.config_Q}\n\n")
                f.write("Vetor de Variâncias de Medição (R Diagonal Config):\n")
                f.write(f"{self.config_R}\n\n")
                f.write("----------------------------------------------------\n\n")

                det_rate = getattr(self, 'detection_rate', 100.0)
                inl_rate = getattr(self, 'inlier_rate', 100.0)

                f.write("--- TAXAS DE DETECÇÃO E CONSISTÊNCIA ---\n")
                f.write(f"Taxa de Detecção (Visível): {det_rate:.2f}%\n")
                f.write(f"Taxa de Inliers (Dentro da Janela): {inl_rate:.2f}%\n\n")

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
                    f.write("RMS X não atingiu convergência.\n")
                if conv_time_y is not None:
                    f.write(f"RMS Y convergiu em {conv_idx_y} frames ({conv_time_y:.2f} s)\n")
                else:
                    f.write("RMS Y não atingiu convergência.\n")
                f.write("(Critério: erro RMS <= 5% do valor final por 10 frames consecutivos)\n\n")

                f.write("--- AVALIAÇÃO DE CONSISTÊNCIA (NIS) ---\n")
                f.write(f"NIS Médio (ideal ≈ {esp_nis} devido a {num_towers} Torres/DOF): {mean_nis_val:.4f}\n")
                f.write(f"Percentual acima do limite de 95% ({chi2_limit}): {nis_pct_above:.2f}%\n")
                f.write(" * Nota: O NIS avalia se a matriz de covariância (P) reflete de forma fidedigna\n")
                f.write("   a real incerteza do modelo. Uma percentagem muito acima de ~5% além do limite\n")
                f.write("   indica subestimação do ruído ou divergência do filtro.\n\n")
                f.write("====================================================\n")

            # ========== EXPORTAÇÃO DOS DADOS BRUTOS EM CSV ==========
            import csv
            csv_path = f"{save_dir}/{video_name}_positions.csv"
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f_csv:
                writer = csv.writer(f_csv, delimiter=',')
                writer.writerow(["Frame", "GT_X(m)", "GT_Y(m)", "Meas_X(m)", "Meas_Y(m)", "Filt_X(m)", "Filt_Y(m)", "NIS"])
                
                for frame_idx in range(len(self.metrics.ground_truth_pts)):
                    gt = self.metrics.ground_truth_pts[frame_idx] if frame_idx < len(self.metrics.ground_truth_pts) else None
                    f_pt = self.metrics.filt_pts[frame_idx] if frame_idx < len(self.metrics.filt_pts) else None
                    m_pt = self.metrics.meas_pts[frame_idx] if frame_idx < len(self.metrics.meas_pts) else (None, None)
                    nis_val = self.metrics.nis_vals[frame_idx] if frame_idx < len(self.metrics.nis_vals) else np.nan
                    
                    gx = f"{gt[0]:.4f}" if gt is not None else ""
                    gy = f"{gt[1]:.4f}" if gt is not None else ""
                    mx = f"{m_pt[0]:.4f}" if m_pt[0] is not None else ""
                    my = f"{m_pt[1]:.4f}" if m_pt[1] is not None else ""
                    fx = f"{f_pt[0]:.4f}" if f_pt is not None else ""
                    fy = f"{f_pt[1]:.4f}" if f_pt is not None else ""
                    nv = f"{nis_val:.4f}" if not np.isnan(nis_val) and not np.isinf(nis_val) else ""
                    
                    writer.writerow([frame_idx, gx, gy, mx, my, fx, fy, nv])

            messagebox.showinfo("Sucesso", f"Análise completa e relatórios exportados com sucesso para:\n{save_dir}/")

        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao gerar relatórios: {error_msg}"))

    def _update_metrics_ui(self):
        """Atualiza as métricas exibidas durante o playback (frame a frame)."""
        if not hasattr(self, 'metrics') or not self.metrics.ground_truth_pts:
            return

        # Ajuste do índice (+1 porque o slice [0:1] pega apenas o elemento 0 do array)
        idx = self.current_frame_idx + 1 
        if idx > len(self.metrics.ground_truth_pts):
            idx = len(self.metrics.ground_truth_pts)

        # 1. Calcula RMSE até o frame atual usando o argumento correto (upto_idx)
        rmse_x, rmse_y = self.metrics.calculate_rmse(upto_idx=idx)

        # 2. Erros assinados até o frame atual
        signed_dx, signed_dy = self.metrics.get_signed_errors(upto_idx=idx)
        mean_err_x = np.mean(signed_dx) if signed_dx else 0.0
        mean_err_y = np.mean(signed_dy) if signed_dy else 0.0

        # 3. NIS até o frame atual
        # Fatiamos a lista de NIS e filtramos NaNs e Infinitos para não quebrar a média
        current_nis_list = self.metrics.nis_vals[:idx]
        valid_nis = [n for n in current_nis_list if not np.isnan(n) and not np.isinf(n)]
        mean_nis = np.mean(valid_nis) if valid_nis else 0.0

        # Define dinamicamente o valor esperado do NIS para a label
        num_towers = len(self.towers) if hasattr(self, 'towers') else 4
        esp_nis = float(num_towers) if num_towers >= 2 else 2.0

        # 4. Atualiza os labels (com segurança na thread principal)
        self.root.after(0, lambda: self.rmse_lbl.config(text=f"RMS (X|Y): {rmse_x:.2f} | {rmse_y:.2f} m"))
        self.root.after(0, lambda: self.mean_err_lbl.config(text=f"E_m (X|Y): {mean_err_x:+.2f} | {mean_err_y:+.2f} m"))
        self.root.after(0, lambda: self.nis_lbl.config(text=f"NIS_m: {mean_nis:.2f} (esp ~{esp_nis:.0f})"))
        
        # --- 5. CÁLCULO E FORMATAÇÃO DO TEMPO DO VÍDEO ---
        fps = getattr(self, 'video_fps', 30.0)
        if fps <= 0: 
            fps = 30.0
        
        current_time_sec = self.current_frame_idx / fps
        total_time_sec = self.total_frames / fps
        
        # Converte segundos para formato MM:SS
        current_min = int(current_time_sec // 60)
        current_sec = int(current_time_sec % 60)
        total_min = int(total_time_sec // 60)
        total_sec = int(total_time_sec % 60)
        
        time_text = f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
        
        self.root.after(0, lambda: self.time_info_lbl.config(text=time_text))

    def _update_plots_ui(self):
        """Delega a atualização dos gráficos ao ChartsDashboard."""
        if hasattr(self, 'dashboard'):
            self.dashboard.update_dashboard(self.metrics, self.current_frame_idx)

    def _style_ax(self, ax):
        """Estiliza os eixos dos gráficos de forma limpa e padronizada."""
        ax.set_facecolor("#f5f5f5")
        ax.tick_params(colors="#666666", labelsize=8)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["left"].set_color("#999999")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _update_ui_metrics_and_complete(self):
        """Finaliza consultando a classe MetricsManager e atualiza todos os indicadores."""
        # Cálculo das métricas finais
        rmse_x, rmse_y = self.metrics.calculate_rmse()
        signed_dx, signed_dy = self.metrics.get_signed_errors()
        
        # Estatísticas
        mean_err_x = np.mean(signed_dx) if signed_dx else 0.0
        mean_err_y = np.mean(signed_dy) if signed_dy else 0.0
        self.detection_rate = 100.0  # ou calcule com base nos raw_z válidos
        self.inlier_rate = 100.0
        
        # NIS médio (ignorando NaNs e Infinitos)
        valid_nis = [n for n in self.metrics.nis_vals if not np.isnan(n) and not np.isinf(n)]
        self.mean_nis = np.mean(valid_nis) if valid_nis else 0.0
        
        # Define dinamicamente o valor esperado do NIS para a label final
        num_towers = len(self.towers) if hasattr(self, 'towers') else 4
        esp_nis = float(num_towers) if num_towers >= 2 else 2.0
        
        # Convergência (exemplo simples - você pode manter o método _find_convergence_frame se existir)
        steady_frame = None
        steady_time = None
        
        if self.metrics.sqerr_x and self.metrics.sqerr_y:
            total_frames = len(self.metrics.sqerr_x)
            
            # Recria o histórico do RMS para os eixos X e Y
            cum_rmse_x = np.sqrt(np.cumsum(self.metrics.sqerr_x) / (np.arange(total_frames) + 1))
            cum_rmse_y = np.sqrt(np.cumsum(self.metrics.sqerr_y) / (np.arange(total_frames) + 1))
            
            # Combina ambos em uma magnitude total de erro: RMS_total = sqrt(RMS_x^2 + RMS_y^2)
            running_rms_total = np.sqrt(cum_rmse_x**2 + cum_rmse_y**2)
            final_val_total = np.sqrt(rmse_x**2 + rmse_y**2)
            
            # Chama a sua função para encontrar o frame de convergência
            # Defina a tolerância (tol) e estabilidade mínima (min_stable) que preferir
            steady_frame = self._find_convergence_frame(running_rms_total, final_val_total, tol=0.05, min_stable=10)
            
            # Se encontrou um frame válido, calcula o tempo em segundos
            if steady_frame is not None and hasattr(self, 'video_fps') and self.video_fps > 0:
                steady_time = steady_frame / self.video_fps
        
        # Atualização dos labels na thread principal
        self.status_lbl.config(text="Status: Processado", fg="#16a34a")
        self.det_rate_lbl.config(text=f"T. Detec : {self.detection_rate:.1f}%")
        self.inlier_rate_lbl.config(text=f"Inliers: {self.inlier_rate:.1f}%")
        self.rmse_lbl.config(text=f"RMS (X|Y): {rmse_x:.2f} | {rmse_y:.2f} m")
        self.mean_err_lbl.config(text=f"E_m (X|Y): {mean_err_x:+.2f} | {mean_err_y:+.2f} m")
        self.nis_lbl.config(text=f"NIS_m: {self.mean_nis:.2f} (esp ~{esp_nis:.0f})")
        
        if steady_time is not None:
            self.steady_state_lbl.config(text=f"Reg. Est.: {steady_time:.1f} s ({steady_frame} fr)")
        else:
            self.steady_state_lbl.config(text="Reg. Est.: -- s (-- fr)")
        
        # Habilita botão de salvar e inicia exibição do vídeo
        self._on_processing_complete()
    
    def _find_convergence_frame(self, running_rms, final_val, tol=0.05, min_stable=10):
        if final_val is None or len(running_rms) < min_stable: return None
        threshold = final_val * (1 + tol)
        start = max(1, len(running_rms) // 5) 
        for i in range(start, len(running_rms) - min_stable + 1):
            if all(v <= threshold for v in running_rms[i:i+min_stable]):
                return i
        return None

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
                self.erro_sensor_lbl.config(text=f"⚡ Erro Dist Simu: {erro_metros:.2f} m ≈ {erro_px:.2f} px")
            except Exception:
                pass

        # 7. Reseta os textos das métricas na UI (Preparando para nova execução)
        if hasattr(self, 'status_lbl'): self.status_lbl.config(text="Status: Carregadas", fg="#16a34a")
        if hasattr(self, 'det_rate_lbl'): self.det_rate_lbl.config(text="Detecção: --%", fg="#4b5563")
        if hasattr(self, 'inlier_rate_lbl'): self.inlier_rate_lbl.config(text="Inliers: --%", fg="#4b5563")
        if hasattr(self, 'rmse_lbl'): self.rmse_lbl.config(text="RMS (X|Y): -- | -- m", fg="#4b5563")
        if hasattr(self, 'mean_err_lbl'): self.mean_err_lbl.config(text="E_m (X|Y): -- | -- m", fg="#4b5563")
        if hasattr(self, 'nis_lbl'): self.nis_lbl.config(text="NIS_m: --", fg="#4b5563")
        if hasattr(self, 'steady_state_lbl'): self.steady_state_lbl.config(text="T_est: -- s (--)", fg="#4b5563")

    def _on_processing_complete(self):
        """Finaliza os cálculos e abre o vídeo salvo para reprodução."""
        self.status_lbl.config(text=f"Status: Concluído | Detecção: {self.inlier_rate:.1f}%")
        self.exec_btn.config(state="normal")
        
        if hasattr(self, 'load_btn'):
            self.load_btn.config(state="normal")

        # === CARREGA O VÍDEO DO HD ===
        if hasattr(self, 'processed_video_path') and os.path.exists(self.processed_video_path):
            self.cap = cv2.VideoCapture(self.processed_video_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0

            # Lê o primeiro frame para exibir a tela cheia e desenhada
            ret, frame = self.cap.read()
            if ret:
                if hasattr(self.tela_viewer, 'display_image'):
                    self.tela_viewer.display_image(frame)
                else:
                    self.tela_viewer.update_image(frame)
                    
                total_time = self.total_frames / self.video_fps
                if hasattr(self, '_format_time'):
                    self.time_info_lbl.config(text=f"00:00 / {self._format_time(total_time)}")

            # Habilita botões do player
            self.prev_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.save_btn.config(state="normal")

            # Atualiza todos os gráficos de uma única vez no final
            if hasattr(self, '_update_plots_ui'):
                self._update_plots_ui()

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
        """Lê o frame do vídeo e atualiza as métricas de forma otimizada para não causar lag."""
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # 1. Atualiza a imagem imediatamente (Super Rápido)
            if hasattr(self.tela_viewer, 'display_image'):
                self.tela_viewer.display_image(frame)
            else:
                self.tela_viewer.update_image(frame)
            
            # 2. Controla o peso computacional (Atualiza gráficos e texto a cada 3 frames)
            self.metrics_update_counter += 1
            if self.metrics_update_counter % 3 == 0 or self.current_frame_idx == self.total_frames - 1:
                # Atualiza Texto (Tempo, FPS, RMSE dinâmico)
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                current_str = self._format_time(self.current_frame_idx / fps)
                total_str = self._format_time(self.total_frames / fps)
                self.time_info_lbl.config(text=f"{current_str} / {total_str}")
                
                self._update_metrics_ui()
                self._update_plots_ui()

    def _poll_playback(self):
        """O Loop que mantém o vídeo rodando no tempo certo (FPS)."""
        if self.playing and self.cap and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_current_frame()
            
            # Ajusta o delay dinamicamente com base no FPS do vídeo gravado
            delay_ms = int(1000 / self.video_fps)
            self.root.after(delay_ms, self._poll_playback)
            
        elif self.playing and self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="⏵")

    def toggle_playback(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "⏵")
        if self.playing:
            self._poll_playback() # Inicia o motor recursivo

    def _format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

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