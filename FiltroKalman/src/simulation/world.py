import numpy as np
from numba import njit
from numpy.linalg import inv

# ====================================================================
# 1. FUNÇÕES MATEMÁTICAS ACELERADAS (NUMBA)
# ====================================================================
@njit
def calc_h_and_H_PVA(x: np.ndarray, bases: np.ndarray):
    """
    Calcula h(x) e a Matriz Jacobiana H para o modelo PVA.
    x: Vetor de estado (6, 1) -> [px, py, vx, vy, ax, ay]
    """
    n_sensores = bases.shape[0]
    hx = np.zeros((n_sensores, 1))
    H = np.zeros((n_sensores, 6)) # 6 colunas
    
    px, py = x[0, 0], x[1, 0]
    
    for i in range(n_sensores):
        bx, by = bases[i, 0], bases[i, 1]
        dx = px - bx
        dy = py - by
        dist = np.sqrt(dx**2 + dy**2)
        
        hx[i, 0] = dist
        
        if dist > 1e-8:
            H[i, 0] = dx / dist
            H[i, 1] = dy / dist
            
    return hx, H

@njit
def get_Q_disc_PVA(q_diag: np.ndarray, dt: float) -> np.ndarray:
    """
    Gera a matriz Q discreta (6x6) para o modelo PVA generalizado.
    q_diag: array com as variâncias contínuas [q1(px), q2(py), q3(vx), q4(vy), q5(ax), q6(ay)]
    dt: delta t (tempo de amostragem)
    """
    q1, q2, q3, q4, q5, q6 = q_diag
    
    Q = np.zeros((6, 6))
    
    # --- Eixo X (Índices 0, 2, 4 correspondentes a px, vx, ax) ---
    Q[0, 0] = q1 * dt + q3 * (dt**3) / 3.0 + q5 * (dt**5) / 20.0
    Q[0, 2] = q3 * (dt**2) / 2.0 + q5 * (dt**4) / 8.0
    Q[2, 0] = Q[0, 2]
    
    Q[0, 4] = q5 * (dt**3) / 6.0
    Q[4, 0] = Q[0, 4]
    
    Q[2, 2] = q3 * dt + q5 * (dt**3) / 3.0
    Q[2, 4] = q5 * (dt**2) / 2.0
    Q[4, 2] = Q[2, 4]
    
    Q[4, 4] = q5 * dt
    
    # --- Eixo Y (Índices 1, 3, 5 correspondentes a py, vy, ay) ---
    Q[1, 1] = q2 * dt + q4 * (dt**3) / 3.0 + q6 * (dt**5) / 20.0
    Q[1, 3] = q4 * (dt**2) / 2.0 + q6 * (dt**4) / 8.0
    Q[3, 1] = Q[1, 3]
    
    Q[1, 5] = q6 * (dt**3) / 6.0
    Q[5, 1] = Q[1, 5]
    
    Q[3, 3] = q4 * dt + q6 * (dt**3) / 3.0
    Q[3, 5] = q6 * (dt**2) / 2.0
    Q[5, 3] = Q[3, 5]
    
    Q[5, 5] = q6 * dt
    
    return Q


# ====================================================================
# 2. ARENA E ENTIDADE (Motor EKF PVA Direto)
# ====================================================================

class World:
    """ Classe para representar a arena e o sensor (medição de distâncias). """
    def __init__(self, bases, noise_std=0.16):
        self.bases = np.array(bases, dtype=float)
        self.num_bases = self.bases.shape[0]
        self.noise_std = float(noise_std)
        
    def measure_distances(self, true_x, true_y):
        measurements = np.zeros((self.num_bases, 1))
        for i in range(self.num_bases):
            bx, by = self.bases[i]
            true_dist = np.sqrt((true_x - bx)**2 + (true_y - by)**2)
            noisy_dist = true_dist + np.random.normal(0, self.noise_std)
            measurements[i, 0] = noisy_dist
        return measurements
    
    def multilaterate(self, distances):
        """
        Estima a posição (x, y) a partir de 3 ou mais distâncias usando Mínimos Quadrados.
        distances: vetor coluna (N, 1) com as medições z.
        Retorna: tupla (est_x, est_y)
        """
        # Precisamos de pelo menos 3 bases para 2D
        if self.num_bases < 3:
            return None, None
            
        A = []
        b = []
        
        # Usaremos a última base como referência (k) para subtrair das outras
        xk, yk = self.bases[-1]
        dk = distances[-1, 0]
        
        for i in range(self.num_bases - 1):
            xi, yi = self.bases[i]
            di = distances[i, 0]
            
            # Matriz A (2 * (xi - xk), 2 * (yi - yk))
            A.append([2 * (xi - xk), 2 * (yi - yk)])
            
            # Vetor b (xi^2 - xk^2 + yi^2 - yk^2 - di^2 + dk^2)
            b_val = (xi**2 - xk**2) + (yi**2 - yk**2) - (di**2) + (dk**2)
            b.append([b_val])
            
        A = np.array(A)
        b = np.array(b)
        
        # Resolve o sistema linear A*P = b usando Mínimos Quadrados
        P_est, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        return float(P_est[0, 0]), float(P_est[1, 0])


class Entidy:
    """
    Motor EKF_PVA puro e gerenciador de estado da entidade rastreada.
    O Filtro de Kalman opera diretamente aqui.
    """
    def __init__(self, dt, bases, q_diag=None, r_diag=None, initial_P=None):
        self.dt = float(dt)
        self.bases = np.array(bases, dtype=float)
        self.num_bases = self.bases.shape[0]

        # 1. Configurando Matriz de Transição F (PVA)
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # 2. Configurando Q (Ruído de Processo) com a função exata de discretização
        if q_diag is not None:
            q_arr = np.array(q_diag, dtype=float)
        else:
            q_arr = np.ones(6) * 1e-2
            
        # UTILIZA A MATRIZ DE DISCRETIZAÇÃO ACELERADA
        self.Q = get_Q_disc_PVA(q_arr, self.dt)

        # 3. Configurando R (Ruído de Medição)
        if r_diag is not None:
            self.R = np.diag(np.array(r_diag, dtype=float))
        else:
            self.R = np.eye(self.num_bases) * 1e-1
        
        # 4. Estado e Covariância Iniciais
        self.P = np.eye(6) * 500.0 if initial_P is None else initial_P.copy()
        self.x = np.zeros((6, 1), dtype=float)
        self.I = np.eye(6)
        
        # Variáveis internas expostas para cálculo de métricas (NIS, etc)
        self.y = None  # Inovação
        self.S = None  # Covariância da Inovação
        
        self.initialized = False

    def initialize(self, initial_x, initial_y):
        """ Inicializa a entidade com uma primeira estimativa (X,Y) """
        self.x[0, 0] = initial_x
        self.x[1, 0] = initial_y
        self.x[2:, 0] = 0.0 # Zera velocidades e acelerações
        self.initialized = True
        
    def predict(self):
        """ Passo 1: Predição do Estado e da Covariância """
        if not self.initialized: return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """ Passo 2: Correção baseada nas medições (z) das Torres """
        if not self.initialized: return
        
        hx, H = calc_h_and_H_PVA(self.x, self.bases)

        z = z.reshape(-1, 1)
        self.y = z - hx
        self.S = H @ self.P @ H.T + self.R
        
        try:
            S_inv = inv(self.S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(self.S)

        K = self.P @ H.T @ S_inv
        
        self.x = self.x + K @ self.y
        
        # Atualização de Joseph (Garante simetria e valores positivos na diagonal de P)
        IKH = self.I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def get_position(self):
        return self.x[:2].reshape(2,)

    def get_uncertainty(self):
        """Retorna o desvio padrão (incerteza) atual para os eixos X e Y."""
        std_x = np.sqrt(max(1e-8, self.P[0, 0]))
        std_y = np.sqrt(max(1e-8, self.P[1, 1]))
        return std_x, std_y