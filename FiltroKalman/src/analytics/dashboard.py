import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.analytics.metrics import MetricsManager

class ChartsDashboard:
    def __init__(self, parent_frame, style_ax_callback=None):
        """
        Gera o painel de gráficos mantendo a estrutura desejada:
        - Linha 1: RMS (X e Y juntos)
        - Linha 2: Histograma | Dispersão (lado a lado)
        - Linha 3: NIS
        """
        self.right_frame = parent_frame
        
        # Define o método de estilização dos eixos (prioriza o original do seu app.py se houver)
        self._style_ax = style_ax_callback if style_ax_callback else self._default_style_ax

        # ==========================================
        # 1. Gráfico Combinado: RMS X e Y
        # ==========================================
        rms_frame = tk.LabelFrame(self.right_frame, text="RMS dos Erros (X e Y)", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, pady=4, borderwidth=1, relief="solid")
        rms_frame.pack(fill="both", expand=True, pady=(0, 4))
        
        self.rms_fig = Figure(figsize=(4.2, 1.8), tight_layout=True, facecolor="white")
        self.rms_ax = self.rms_fig.add_subplot(111)
        self._style_ax(self.rms_ax)
        self.rms_canvas = FigureCanvasTkAgg(self.rms_fig, master=rms_frame)
        self.rms_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ==========================================
        # Container do Meio: Histograma | Dispersão
        # ==========================================
        middle_container = tk.Frame(self.right_frame, bg="white")
        middle_container.pack(fill="both", expand=True, pady=(0, 4))
        middle_container.columnconfigure(0, weight=1)
        middle_container.columnconfigure(1, weight=1)

        # 2. Histograma Frame (Esquerda)
        hist_frame = tk.LabelFrame(middle_container, text="Histograma", 
                                  font=("Segoe UI", 9, "bold"), bg="white", fg="#333333",
                                  padx=2, pady=2, borderwidth=1, relief="solid")
        hist_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        
        self.hist_fig = Figure(figsize=(2.1, 1.8), tight_layout=True, facecolor="white")
        self.hist_ax = self.hist_fig.add_subplot(111)
        self._style_ax(self.hist_ax)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # 3. Scatter Frame / Dispersão (Direita)
        scatter_frame = tk.LabelFrame(middle_container, text="Dispersão", 
                                  font=("Segoe UI", 9, "bold"), bg="white", fg="#333333",
                                  padx=2, pady=2, borderwidth=1, relief="solid")
        scatter_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        
        self.scatter_fig = Figure(figsize=(2.1, 1.8), tight_layout=True, facecolor="white")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self._style_ax(self.scatter_ax)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=scatter_frame)
        self.scatter_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ==========================================
        # 4. Gráfico do NIS (Normalized Innovation Squared)
        # ==========================================
        nis_frame = tk.LabelFrame(self.right_frame, text="Histórico de Atualização do NIS", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, pady=4, borderwidth=1, relief="solid")
        nis_frame.pack(fill="both", expand=True, pady=(0, 0))
        
        self.nis_fig = Figure(figsize=(4.2, 1.8), tight_layout=True, facecolor="white")
        self.nis_ax = self.nis_fig.add_subplot(111)
        self._style_ax(self.nis_ax)
        self.nis_canvas = FigureCanvasTkAgg(self.nis_fig, master=nis_frame)
        self.nis_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _default_style_ax(self, ax):
        """Caso o app principal não passe o método self._style_ax, aplica este padrão simples."""
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)

    def draw_all(self):
        """Redesenha e atualiza todos os canvas na interface do Tkinter."""
        self.rms_canvas.draw_idle()
        self.hist_canvas.draw_idle()
        self.scatter_canvas.draw_idle()
        self.nis_canvas.draw_idle()

    def update_dashboard(self, metrics: MetricsManager, upto_idx: int):
        """
        Lê os dados do MetricsManager e atualiza os subplots até o frame atual.
        """
        if upto_idx <= 0 or not metrics.sqerr_x:
            return

        # 1. Atualização do RMS (X e Y juntos)
        self.rms_ax.clear()
        self._style_ax(self.rms_ax) 
        cum_rmse_x = np.sqrt(np.cumsum(metrics.sqerr_x[:upto_idx]) / (np.arange(upto_idx) + 1))
        cum_rmse_y = np.sqrt(np.cumsum(metrics.sqerr_y[:upto_idx]) / (np.arange(upto_idx) + 1))
        
        self.rms_ax.plot(cum_rmse_x, color='#2563eb', linewidth=1.5, label='RMS X')
        self.rms_ax.plot(cum_rmse_y, color='#ea580c', linewidth=1.5, label='RMS Y')
        
        self.rms_ax.set_xlabel("Frame", fontsize=8)
        self.rms_ax.set_ylabel("RMSE", fontsize=8)
        self.rms_ax.legend(loc='upper right', fontsize=7, facecolor="#f5f5f5", edgecolor="#999999")

        # Extrai os erros assinados (resíduos) para o histograma e dispersão
        signed_dx, signed_dy = metrics.get_signed_errors(upto_idx)

        # 2. Atualização do Histograma
        self.hist_ax.clear()
        self.hist_ax.set_title("Histograma de Erros", fontsize=8, fontweight="bold", alpha=0.7)
        self._style_ax(self.hist_ax)
        if signed_dx and signed_dy:
            self.hist_ax.hist(signed_dx, bins=15, alpha=0.6, color='#3b82f6', label='Erro X')
            self.hist_ax.hist(signed_dy, bins=15, alpha=0.6, color='#f97316', label='Erro Y')
            
            self.hist_ax.set_xlabel("Magnitude do Erro", fontsize=8)
            self.hist_ax.set_ylabel("Frequência", fontsize=8)
            self.hist_ax.legend(loc='upper right', fontsize=7, facecolor="#f5f5f5", edgecolor="#999999")

        # 3. Atualização do Scatter Plot (Dispersão)
        self.scatter_ax.clear()
        self._style_ax(self.scatter_ax)
        if signed_dx and signed_dy:
            self.scatter_ax.scatter(signed_dx, signed_dy, alpha=0.5, c='purple', edgecolors='k', s=15)
            self.scatter_ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            self.scatter_ax.axvline(0, color='black', linewidth=1, alpha=0.5)
            
            self.scatter_ax.set_xlabel("Erro X", fontsize=8)
            self.scatter_ax.set_ylabel("Erro Y", fontsize=8)
            
            max_err = max(max(np.abs(signed_dx)), max(np.abs(signed_dy)), 0.1)
            self.scatter_ax.set_xlim(-max_err * 1.2, max_err * 1.2)
            self.scatter_ax.set_ylim(-max_err * 1.2, max_err * 1.2)

        # =========================================================================
        # 4. ATUALIZAÇÃO DO NIS (Alterado para STEM PLOT)
        # =========================================================================
        self.nis_ax.clear()
        self._style_ax(self.nis_ax)
        
        if hasattr(metrics, 'nis_vals') and metrics.nis_vals:
            nis_data = metrics.nis_vals[:upto_idx]
            
            # Filtra os índices válidos para ignorar NaNs/Infs que quebram o gráfico
            indices_validos = [i for i, v in enumerate(nis_data) if not np.isnan(v) and not np.isinf(v)]
            xf = indices_validos
            yf = [nis_data[i] for i in indices_validos]
            
            if yf:
                # --- MODIFICAÇÃO AQUI: Uso do stem ---
                # Criamos o plot e capturamos os componentes para estilização
                # Usamos plt.setp (set property) que é mais eficiente no OO
                markerline, stemlines, baseline = self.nis_ax.stem(xf, yf)

                # Estilização profissional do Stem (Verde #16a34a)
                # 1. Marcadores (as bolinhas no topo): Pequenas e opacas
                plt.setp(markerline, marker='o', markersize=3, color='#16a34a', alpha=0.8, label='NIS')
                
                # 2. Hastes (linhas verticais): Finas e semi-transparentes para não poluir
                plt.setp(stemlines, color='#16a34a', linewidth=0.8, alpha=0.4)
                
                # 3. Linha de base (no y=0): Ocultamos para limpar o visual, já que o eixo X já existe
                plt.setp(baseline, visible=False)
                
                # Linha de referência padrão para 4 Torres (Limite Chi-Quadrado 95% = 9.488)
                chi2_limit = 9.488
                self.nis_ax.axhline(chi2_limit, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'Lim. 95% ({chi2_limit})')
                
                self.nis_ax.set_xlabel("Frame", fontsize=8)
                self.nis_ax.set_ylabel("Valor NIS", fontsize=8)
                self.nis_ax.legend(loc='upper right', fontsize=7, facecolor="#f5f5f5", edgecolor="#999999")
                
                # Ajuste de limite dinâmico para evitar picos esmagando a escala do gráfico
                teto_visual = float(np.percentile(yf, 95) * 1.8) if len(yf) > 10 else chi2_limit * 1.5
                self.nis_ax.set_ylim(0, max(chi2_limit * 1.5, teto_visual))
                self.nis_ax.set_xlim(0, max(30, upto_idx * 1.02))

        # 5. Redesenha tudo de forma otimizada
        self.draw_all()