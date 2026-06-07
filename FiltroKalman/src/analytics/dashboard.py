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

    def update_dashboard(self, metrics: MetricsManager, upto_idx: int):
        """
        Lê os dados do MetricsManager e atualiza os 4 subplots até o frame atual.
        """
        if upto_idx <= 0 or not metrics.sqerr_x:
            return

        # 1. Atualização do RMS X
        self.rmsx_ax.clear()
        self._style_ax(self.rmsx_ax) # Reaplica o estilo após limpar
        cum_rmse_x = np.sqrt(np.cumsum(metrics.sqerr_x[:upto_idx]) / (np.arange(upto_idx) + 1))
        self.rmsx_ax.plot(cum_rmse_x, color='#2563eb', linewidth=1.5)

        # 2. Atualização do RMS Y
        self.rmsy_ax.clear()
        self._style_ax(self.rmsy_ax)
        cum_rmse_y = np.sqrt(np.cumsum(metrics.sqerr_y[:upto_idx]) / (np.arange(upto_idx) + 1))
        self.rmsy_ax.plot(cum_rmse_y, color='#ea580c', linewidth=1.5)

        # Extrai os erros assinados (resíduos) para o histograma e dispersão
        signed_dx, signed_dy = metrics.get_signed_errors(upto_idx)

        # 3. Atualização do Histograma
        self.hist_ax.clear()
        self._style_ax(self.hist_ax)
        if signed_dx and signed_dy:
            self.hist_ax.hist(signed_dx, bins=15, alpha=0.6, color='#3b82f6', label='Erro X')
            self.hist_ax.hist(signed_dy, bins=15, alpha=0.6, color='#f97316', label='Erro Y')
            self.hist_ax.legend(loc='upper right', fontsize=7, facecolor="#f5f5f5", edgecolor="#999999")

        # 4. Atualização do Scatter Plot (Dispersão)
        self.scatter_ax.clear()
        self._style_ax(self.scatter_ax)
        if signed_dx and signed_dy:
            self.scatter_ax.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=15)
            self.scatter_ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            self.scatter_ax.axvline(0, color='black', linewidth=1, alpha=0.5)
            
            # Ajusta limites dinamicamente
            max_err = max(max(np.abs(signed_dx)), max(np.abs(signed_dy)), 0.1)
            self.scatter_ax.set_xlim(-max_err * 1.2, max_err * 1.2)
            self.scatter_ax.set_ylim(-max_err * 1.2, max_err * 1.2)

        # 5. Redesenha tudo
        self.draw_all()