import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.analytics.metrics import MetricsManager

class ChartsDashboard:
    def __init__(self, parent_frame, style_ax_callback=None):
        """
        Gera o painel de gráficos mantendo a estrutura desejada com descrições breves.
        """
        self.right_frame = parent_frame
        
        # Define o método de estilização dos eixos
        self._style_ax = style_ax_callback if style_ax_callback else self._default_style_ax

        # Cores e fontes para as descrições
        desc_font = ("Segoe UI", 8, "italic")
        desc_color = "#555555"

        # ==========================================
        # 1. Gráfico Combinado: RMS X e Y
        # ==========================================
        rms_frame = tk.LabelFrame(self.right_frame, text="RMS dos Erros (X e Y)", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, borderwidth=1, relief="solid")
        rms_frame.pack(fill="both", expand=True, pady=(0, 4))
        
        rms_desc = tk.Label(rms_frame, text="Mede a precisão acumulada: RMSE = √(Σ e² / N). Menores valores indicam maior exatidão.",
                            font=desc_font, fg=desc_color, bg="white", justify="left", anchor="w", wraplength=380)
        rms_desc.pack(fill="x", side="top", pady=(0, 2))
        
        self.rms_fig = Figure(figsize=(4.2, 1.6), tight_layout=True, facecolor="white")
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
                                  padx=2, borderwidth=1, relief="solid")
        hist_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        
        hist_desc = tk.Label(hist_frame, text="Frequência dos resíduos. Espera-se perfil Gaussiano centrado em zero.",
                             font=desc_font, fg=desc_color, bg="white", justify="left", anchor="w", wraplength=180)
        hist_desc.pack(fill="x", side="top", pady=(0, 2))
        
        self.hist_fig = Figure(figsize=(2.1, 1.6), tight_layout=True, facecolor="white")
        self.hist_ax = self.hist_fig.add_subplot(111)
        self._style_ax(self.hist_ax)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        # 3. Scatter Frame / Dispersão (Direita)
        scatter_frame = tk.LabelFrame(middle_container, text="Dispersão", 
                                  font=("Segoe UI", 9, "bold"), bg="white", fg="#333333",
                                  padx=2, borderwidth=1, relief="solid")
        scatter_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        
        scatter_desc = tk.Label(scatter_frame, text="Erros em 2D. Revela tendências espaciais ou desalinhamentos.",
                                font=desc_font, fg=desc_color, bg="white", justify="left", anchor="w", wraplength=180)
        scatter_desc.pack(fill="x", side="top", pady=(0, 2))
        
        self.scatter_fig = Figure(figsize=(2.1, 1.6), tight_layout=True, facecolor="white")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self._style_ax(self.scatter_ax)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=scatter_frame)
        self.scatter_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ==========================================
        # 4. Gráfico do NIS (Normalized Innovation Squared)
        # ==========================================
        nis_frame = tk.LabelFrame(self.right_frame, text="Histórico de Atualização do NIS", 
                                  font=("Segoe UI", 10, "bold"), bg="white", fg="#333333",
                                  padx=4, borderwidth=1, relief="solid")
        nis_frame.pack(fill="both", expand=True, pady=(0, 0))
        
        nis_desc = tk.Label(nis_frame, text="Consistência (95%): NIS = νᵀ S⁻¹ ν. Manter dentro da faixa verde é o ideal. Fora indica erro super/subestimado.",
                            font=desc_font, fg=desc_color, bg="white", justify="left", anchor="w", wraplength=380)
        nis_desc.pack(fill="x", side="top", pady=(0, 0))
        
        self.nis_fig = Figure(figsize=(4.2, 2.2), tight_layout=True, facecolor="white")
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
        # 4. ATUALIZAÇÃO DO NIS (Alterado para Intervalo Mínimo e Máximo)
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
                markerline, stemlines, baseline = self.nis_ax.stem(xf, yf)

                # Estilização do Stem (Verde #16a34a)
                plt.setp(markerline, marker='o', markersize=3, color='#16a34a', alpha=0.8, label='NIS')
                plt.setp(stemlines, color='#16a34a', linewidth=0.8, alpha=0.4)
                plt.setp(baseline, visible=False)
                
                # [ALTERADO] Identificação dinâmica de DoF e definição dos limites Bicaudais (95%)
                num_towers = getattr(metrics, 'num_towers', 4)
                if num_towers == 4:
                    chi2_lower = 0.484
                    chi2_upper = 11.143
                elif num_towers == 3:
                    chi2_lower = 0.216
                    chi2_upper = 9.348
                else:
                    chi2_lower = 0.051
                    chi2_upper = 7.378
                
                # Adiciona as duas linhas de referência (mínima e máxima)
                self.nis_ax.axhline(chi2_upper, color='red', linestyle='--', linewidth=1.0, alpha=0.8, label=f'Sup 97.5% ({chi2_upper})')
                self.nis_ax.axhline(chi2_lower, color='orange', linestyle='--', linewidth=1.0, alpha=0.8, label=f'Inf 2.5% ({chi2_lower})')
                
                # Adiciona a faixa preenchida verde representando a área de consistência ideal
                self.nis_ax.axhspan(chi2_lower, chi2_upper, color='green', alpha=0.06, label='Consistente')
                
                self.nis_ax.set_xlabel("Frame", fontsize=8)
                self.nis_ax.set_ylabel("Valor NIS", fontsize=8)
                self.nis_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),                ncol=2, 
                                   fontsize=7, facecolor="#f5f5f5", edgecolor="#999999", 
                                   framealpha=0.9)
                
                # Ajuste de limite dinâmico com base no limite superior para evitar achatamento visual por picos
                teto_visual = float(np.percentile(yf, 95) * 1.8) if len(yf) > 10 else chi2_upper * 1.5
                self.nis_ax.set_ylim(0, max(chi2_upper * 1.6, teto_visual))
                self.nis_ax.set_xlim(0, max(30, upto_idx * 1.02))

        # 5. Redesenha tudo de forma otimizada
        self.draw_all()

    def plot_final_results(self, metrics: MetricsManager):
        """
        Plota imediatamente todo o histórico da simulação de uma só vez.
        Útil para visualizar o resultado final sem precisar reproduzir o vídeo.
        """
        if not metrics or not metrics.sqerr_x:
            return
            
        total_frames = len(metrics.sqerr_x)
        self.update_dashboard(metrics, total_frames)