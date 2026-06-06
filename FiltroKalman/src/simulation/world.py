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
        # Velocidades e acelerações não afetam diretamente a medição da distância, ficam em 0.

    return hx, H


# ====================================================================
# 2. MOTOR EKF PVA PURO
# ====================================================================

class EKF_PVA:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Inicializa o Filtro de Kalman Estendido PVA otimizado.
        """
        self.x = x0.copy()  # Vetor de estado (6x1)
        self.P = P0.copy()  # Covariância do erro (6x6)
        self.F = F          # Matriz de transição discreta (6x6)
        self.Q = Q          # Covariância do processo (6x6)
        self.R = R          # Covariância da medição (NxN)
        self.I = np.eye(6)  # Matriz Identidade para o Update
        
    def predict(self):
        """ Passo 1: Predição do Estado e da Covariância """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray, bases: np.ndarray):
        """ Passo 2: Correção baseada nas medições (z) """
        hx, H = calc_h_and_H_PVA(self.x, bases)

        z = z.reshape(-1, 1) # Garante que z seja um vetor coluna
        y = z - hx
        
        S = H @ self.P @ H.T + self.R
        
        # Para evitar problemas de singularidade com numpy.inv em ambientes com muito ruído,
        # utilizamos pseudoinversa np.linalg.pinv caso não use o Numba dentro deste passo
        try:
            S_inv = inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ H.T @ S_inv
        
        self.x = self.x + K @ y
        
        # Atualização de Joseph para garantir estabilidade numérica da matriz de covariância
        # (Substituindo self.P = (self.I - K @ H) @ self.P para maior robustez)
        IKH = self.I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T


# ====================================================================
# 3. WORLD E ENTIDY (Integração com a GUI)
# ====================================================================

class World:
    """
    Classe para representar a arena de simulação e atuar como o sensor do sistema.
    """
    def __init__(self, bases, noise_std=0.16):
        self.bases = np.array(bases, dtype=float)
        self.num_bases = self.bases.shape[0]
        self.noise_std = float(noise_std)
        
    def measure_distances(self, true_x, true_y):
        """ Mede a distância Euclidiana adicionando ruído Gaussiano """
        measurements = np.zeros((self.num_bases, 1))
        for i in range(self.num_bases):
            bx, by = self.bases[i]
            true_dist = np.sqrt((true_x - bx)**2 + (true_y - by)**2)
            noisy_dist = true_dist + np.random.normal(0, self.noise_std)
            measurements[i, 0] = noisy_dist
        return measurements


class Entidy:
    """
    Wrapper da GUI para gerenciar o estado da entidade rastreada através do motor EKF_PVA.
    """
    def __init__(self, dt, bases, q_diag=None, r_diag=None, initial_P=None):
        self.dt = float(dt)
        self.bases = np.array(bases, dtype=float)
        self.num_bases = self.bases.shape[0]

        # 1. Configurando Matriz de Transição F (PVA)
        F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # 2. Configurando Q
        if q_diag is not None:
            Q = np.diag(np.array(q_diag, dtype=float))
        else:
            Q = np.eye(6) * 1e-2
        Qd = Q * self.dt

        # 3. Configurando R
        if r_diag is not None:
            R = np.diag(np.array(r_diag, dtype=float))
        else:
            R = np.eye(self.num_bases) * 1e-1
        
        # 4. Configurando P Inicial
        P0 = np.eye(6) * 500.0 if initial_P is None else initial_P
        
        # O estado inicial será 0, ele deve ser sobreescrito pelo método 'initialize'
        x0 = np.zeros((6, 1), dtype=float)

        # Instancia o motor EKF
        self.ekf = EKF_PVA(x0, P0, F, Qd, R)
        self.initialized = False

    def initialize(self, initial_x, initial_y):
        """ Inicializa a entidade com uma primeira estimativa (X,Y) """
        self.ekf.x[0, 0] = initial_x
        self.ekf.x[1, 0] = initial_y
        self.ekf.x[2, 0] = 0.0  # vx
        self.ekf.x[3, 0] = 0.0  # vy
        self.ekf.x[4, 0] = 0.0  # ax
        self.ekf.x[5, 0] = 0.0  # ay
        self.initialized = True
        
    def predict(self):
        self.ekf.predict()

    def update(self, meas):
        if not self.initialized:
            return 
        self.ekf.update(meas, self.bases)

    def get_state(self):
        return self.ekf.x.copy()

    def get_position(self):
        return self.ekf.x[:2].reshape(2,)

    def get_uncertainty(self):
        """Retorna o desvio padrão (incerteza) atual para os eixos X e Y."""
        std_x = np.sqrt(self.ekf.P[0, 0])
        std_y = np.sqrt(self.ekf.P[1, 1])
        return std_x, std_y

    def print_summary(self):
        if not self.initialized:
            print("[Entidy] Status: Não inicializada")
            return
            
        x, y = self.get_position()
        vx, vy = self.ekf.x[2, 0], self.ekf.x[3, 0]
        std_x, std_y = self.get_uncertainty()
        vel_tot = np.sqrt(vx**2 + vy**2)

        print(f" [EKF] Pos: ({x:6.2f}, {y:6.2f}) | Vel: {vel_tot:6.2f}m/s | Incerteza(X,Y): ({std_x:5.2f}, {std_y:5.2f})")