import numpy as np
import csv

class MetricsManager:
    def __init__(self):
        self.clear()

    def clear(self):
        """Reinicializa todos os históricos de dados."""
        self.ground_truth_pts = []  # Lista de tuplas (x, y) reais
        self.filt_pts = []          # Lista de tuplas (x, y) estimadas pelo Kalman
        self.meas_pts = []          # NOVO: Posições detectadas (com ruído) calculadas via multilateração
        self.sqerr_x = []           # Erros quadráticos em X
        self.sqerr_y = []           # Erros quadráticos em Y
        self.nis_vals = []          # Valores de NIS (Normalized Innovation Squared)
        self.measurements_raw = []  # Leituras brutas das distâncias
        self.P_mats = []            # Matrizes de Covariância do EKF

    def push_frame(self, gt_x, gt_y, est_x, est_y, meas_x=None, meas_y=None, nis_val=np.nan, raw_z=None, P_mat=None):
        """Adiciona os dados de um frame."""
        self.ground_truth_pts.append((gt_x, gt_y))
        self.filt_pts.append((est_x, est_y))
        self.meas_pts.append((meas_x, meas_y)) # Guarda o ponto ruidoso da detecção (bolinha vermelha)
        
        self.sqerr_x.append((est_x - gt_x) ** 2)
        self.sqerr_y.append((est_y - gt_y) ** 2)
        self.measurements_raw.append(raw_z)
        self.P_mats.append(np.array(P_mat, copy=True) if P_mat is not None else None)
        
        # O NIS verdadeiro (Inovação) agora é calculado no app.py e passado para cá.
        self.nis_vals.append(nis_val)

    def get_signed_errors(self, upto_idx=None):
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
        if upto_idx is None:
            upto_idx = len(self.sqerr_x)
            
        if upto_idx == 0:
            return 0.0, 0.0
            
        rmse_x = np.sqrt(np.mean(self.sqerr_x[:upto_idx]))
        rmse_y = np.sqrt(np.mean(self.sqerr_y[:upto_idx]))
        return float(rmse_x), float(rmse_y)

    def save_to_csv(self, filepath):
        if not self.ground_truth_pts:
            return False
            
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Atualizado para incluir as coordenadas medidas (Multilateração) no CSV
            writer.writerow([
                "Frame", "GT_X(m)", "GT_Y(m)", "Meas_X(m)", "Meas_Y(m)", "Est_X(m)", "Est_Y(m)", 
                "SqErr_X(m2)", "SqErr_Y(m2)", "NIS"
            ])
            
            for i in range(len(self.ground_truth_pts)):
                gt = self.ground_truth_pts[i]
                est = self.filt_pts[i]
                meas = self.meas_pts[i]
                
                mx = meas[0] if meas[0] is not None else ""
                my = meas[1] if meas[1] is not None else ""
                
                writer.writerow([
                    i, gt[0], gt[1], mx, my, est[0], est[1], 
                    self.sqerr_x[i], self.sqerr_y[i], self.nis_vals[i]
                ])
        return True