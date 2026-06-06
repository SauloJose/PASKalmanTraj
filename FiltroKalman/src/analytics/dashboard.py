import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.analytics.metrics import MetricsManager

class ChartsDashboard:
    def __init__(self, parent_frame, style_ax_callback=None):
        """
        Gera o painel de gráficos mantendo a estrutura exata do _build_right_painel original.
        
        :param parent_frame: O frame do Tkinter onde os gráficos serão inseridos (self.right_frame)
        :param style_ax_callback: Referência opcional para o método self._style_ax da classe principal.
                                 Se não for fornecido, usa um estilizador padrão seguro interno.
        """
        self.right_frame = parent_frame
        
        # Define o método de estilização dos eixos (prioriza o original do seu app.py se houver)
        self._style_ax = style_ax_callback if style_ax_callback else self._default_style_ax

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

    def _default_style_ax(self, ax):
        """Caso o app principal não passe o método self._style_ax, aplica este padrão simples."""
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)

    def draw_all(self):
        """Redesenha e atualiza todos os canvas na interface do Tkinter."""
        self.rmsx_canvas.draw_idle()
        self.rmsy_canvas.draw_idle()
        self.hist_canvas.draw_idle()
        self.scatter_canvas.draw_idle()