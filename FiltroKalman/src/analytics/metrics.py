import numpy as np
import csv

class MetricsManager:
    def __init__(self):
        self.clear()

    def clear(self):
        """Reinicializa todos os históricos de dados."""
        self.ground_truth_pts = []  # Lista de tuplas (x, y) reais
        self.filt_pts = []          # Lista de tuplas (x, y) estimadas pelo Kalman
        self.sqerr_x = []           # Erros quadráticos em X
        self.sqerr_y = []           # Erros quadráticos em Y
        self.nis_vals = []          # Valores de NIS (Normalized Innovation Squared)
        self.measurements_raw = []  # Leituras brutas das distâncias das torres

    def push_frame(self, gt_x, gt_y, est_x, est_y, nis_val=0.0, raw_z=None):
        """Adiciona os dados calculados de um frame específico."""
        self.ground_truth_pts.append((gt_x, gt_y))
        self.filt_pts.append((est_x, est_y))
        
        # Erros lineares e quadráticos
        self.sqerr_x.append((est_x - gt_x) ** 2)
        self.sqerr_y.append((est_y - gt_y) ** 2)
        self.nis_vals.append(nis_val)
        
        if raw_z is not None:
            self.measurements_raw.append(raw_z)

    def get_signed_errors(self, upto_idx=None):
        """Retorna os erros com sinal (erros residuais) até um certo frame."""
        if upto_idx is None:
            upto_idx = len(self.ground_truth_pts)
            
        gt = np.array(self.ground_truth_pts[:upto_idx])
        est = np.array(self.filt_pts[:upto_idx])
        
        if len(gt) == 0:
            return [], []
            
        dx = est[:, 0] - gt[:, 0]
        dy = est[:, 1] - gt[:, 1]
        return dx.tolist(), dy.tolist()

    def calculate_rmse(self, upto_idx=None):
        """Calcula o RMSE (Root Mean Squared Error) acumulado em metros."""
        if upto_idx is None:
            upto_idx = len(self.sqerr_x)
            
        if upto_idx == 0:
            return 0.0, 0.0
            
        rmse_x = np.sqrt(np.mean(self.sqerr_x[:upto_idx]))
        rmse_y = np.sqrt(np.mean(self.sqerr_y[:upto_idx]))
        return float(rmse_x), float(rmse_y)

    def save_to_csv(self, filepath):
        """Exporta de forma limpa e tabular todo o histórico para análise externa."""
        if not self.ground_truth_pts:
            return False
            
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Cabeçalho acadêmico
            writer.writerow([
                "Frame", "GT_X(m)", "GT_Y(m)", "Est_X(m)", "Est_Y(m)", 
                "SqErr_X(m2)", "SqErr_Y(m2)", "NIS"
            ])
            
            for i in range(len(self.ground_truth_pts)):
                gt = self.ground_truth_pts[i]
                est = self.filt_pts[i]
                writer.writerow([
                    i, gt[0], gt[1], est[0], est[1], 
                    self.sqerr_x[i], self.sqerr_y[i], self.nis_vals[i]
                ])
        return True