"""
    Esse arquivo aqui é responsável por simular o mundo real, atribuindo lógica a imagem e transformando os pixels em coordenadas reais
    em metros.

"""
import numpy as np

class World:
    """
        Classe para representar as coordenadas do mundo
    """
    def __init__(self, dimPX, dimPY, dimPOX, dimPOY):
        # Dimensões para conversão 
        self._dPX = dimPX
        self._dPY = dimPY
        self._dPOX = dimPOX
        self._dPOY = dimPOY

        # Origem do sistema de mundo 
        # o extremo inferior esquerdo será o centro do mundo nesse sistema
        self._POri = [0,0]
        self._POriPx = [0, dimPY]

        # proporção px / cm
        self.prop_px_per_m = dimPX / dimPOX 

        # Possível implementação futura
        self.entidies: list[Entidy] = None 

    def img2world(self, px, py):
        """
        Retorna as coordenadas (X,Y) em emtros no mundo
        """
        # X cresce para a direita (mesmo sentido do pixel)
        x_m = px * (self._dPOX / self._dPX)
        # Y cresce para cima (invertido: a base da imagem dimPY vira o 0 do mundo)
        y_m = (self._dPY - py) * (self._dPOY / self._dPY)
        return x_m, y_m

    def world2img(self, x, y):
        """
        Retorna as coordenadsa em pixels do objeto na imagem.
        """
        # Inversão matemática da função acima
        px = x * (self._dPX / self._dPOX)
        py = self._dPY - (y * (self._dPY / self._dPOY))
        return int(px), int(py)



# Classe para representar uma entidade do detector
class Entidy:
    """
        Essa classe representa uma entidade do sistema de medição
        Nela o modelo de movimento será implementado

        Essa entidade é modelada com o modelo de 6 estados
        [x,y,vx,vy,ax,ay]
    """
    def __init__(self, dt, q_diag=None, r_diag=None, initial_P = None):
        self.dt = float(dt)

        # Matrix de transiçã ode estados já discretizada
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix (we measure x, y only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)

        # Covariância do processo Q
        if q_diag is not None:
            self.Q = np.diag(np.array(q_diag, dtype=float))
        else:
            self.Q = np.eye(6) * 1e-2
        
        # Covariância discretizada do Processo 
        # Essa aproximação é devido ao pequeno tempo de propagação dos erros.
        self.Qd = self.Q * self.dt


        # Measurement covariance
        if r_diag is not None:
            self.R = np.diag(np.array(r_diag, dtype=float))
        else:
            self.R = np.eye(2) * 1e-1
        
        # State covariance
        self.P = np.eye(6) * 500.0 if initial_P == None else initial_P
        
        # State vector
        self.x = np.zeros((6, 1), dtype=float)
        self.initialized = False

    def initialize(self, meas):
        """
            Inicializo o estado com a primeira medição
        """
        m = np.asarray(meas).reshape(2,1)
        
        # Adicione o ", 0" para extrair o valor escalar exato (0 dimensões)
        self.x[0, 0] = float(m[0, 0])  
        self.x[1, 0] = float(m[1, 0])  
        
        self.x[2, 0] = 0.0  # vx
        self.x[3, 0] = 0.0  # vy
        self.x[4, 0] = 0.0  # ax
        self.x[5, 0] = 0.0  # ay
        self.initialized = True
        
    # Predição da posição da entidade
    def predict(self):
        # Atualizando a crença a priori da entidade (belief)
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Qd

    # Atualização da posição da entidade
    def update(self, meas):
        z = np.asarray(meas).reshape(2,1)

        # Se não tiver sido inicializado anteriormente, primeiro inicializo
        if not self.initialized:
            self.initialize(z)
            return 
        
        # Inovação
        y = z - self.H.dot(self.x)

        # covariância da Inovação 
        S = self.H.dot(self.P).dot(self.H.T) + self.R

        # Ganho de Kalman (garante a minimização do erro quadrático médio)
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        ## Atualização da Crença a posteriori da entidade belief()
        # Atualização do estado e da predição (u_x)
        self.x = self.x + K.dot(y)
        
        # Atualização da covariância interna do estado
        self.P = (np.eye(6) - K.dot(self.H)).dot(self.P)


    def get_state(self):
        """ Retorno o vetor de estados completos"""
        return self.x.copy()

    
    def get_position(self):
        """ Puxo a posição [x,y]"""
        return self.x[:2].reshape(2,)

    # --- NOVOS MÉTODOS ADICIONADOS ---

    def __str__(self):
        """Retorna uma linha de resumo rápido do estado atual do objeto."""
        if not self.initialized:
            return "[Entidy] Status: Não inicializada"
        pos = self.get_position()
        vel = self.x[2:4].reshape(2,)
        return f"[Entidy] Posição: ({pos[0]:.2f}, {pos[1]:.2f}) | Velocidade: ({vel[0]:.2f}, {vel[1]:.2f})"

    def get_uncertainty(self):
        """Retorna o desvio padrão (incerteza) atual para os eixos X e Y."""
        # A incerteza é a raiz quadrada da variância (elementos da diagonal de P)
        std_x = np.sqrt(self.P[0, 0])
        std_y = np.sqrt(self.P[1, 1])
        return std_x, std_y

    def print_summary(self, world=None):
        """
        Imprime no console um relatório completo, limpo e estruturado dos dados 
        cinemáticos e estatísticos obtidos pelo Filtro de Kalman.
        """
        print("\n" + "="*55)
        print("               RELATÓRIO DE TELEMETRIA (KALMAN)        ")
        print("="*55)
        
        if not self.initialized:
            print(" STATUS: Aguardando primeira medição do sensor...")
            print("="*55)
            return

        # Captura de variáveis individuais
        x, y = self.get_position()
        vx, vy = self.x[2, 0], self.x[3, 0]
        ax, ay = self.x[4, 0], self.x[5, 0]
        std_x, std_y = self.get_uncertainty()

        # Cálculos de Intensidade Resultante (Magnitude Vetorial)
        velocidade_total = np.sqrt(vx**2 + vy**2)
        aceleracao_total = np.sqrt(ax**2 + ay**2)

        print(f" STATUS: Ativo e Rastreado")
        print(f" • Posição Real:     X = {x:7.3f} m  |  Y = {y:7.3f} m")
        print(f" • Incerteza (±σ):   X = {std_x:7.3f} m  |  Y = {std_y:7.3f} m")
        print(f" • Velocidade:      Vx = {vx:7.3f} m/s | Vy = {vy:7.3f} m/s")
        print(f"   --> Vetor Velocidade Resultante: {velocidade_total:.3f} m/s")
        print(f" • Aceleração:      Ax = {ax:7.3f} m/s²| Ay = {ay:7.3f} m/s²")
        print(f"   --> Vetor Aceleração Resultante: {aceleracao_total:.3f} m/s²")

        # Integração inteligente com a classe World
        if world is not None:
            print("-"*55)
            try:
                px, py = world.world2img(x, y)
                print(f" • Mapeamento de Tela: Equivalente ao Pixel ({px}, {py})")
            except Exception as e:
                print(f" [Aviso] Não foi possível converter para pixels: {e}")
                
        print("="*55)


# Classe para representar uma entidade do detector
class Robot:
    """
        Essa classe representa uma entidade do sistema de medição
        Nela o modelo de movimento será implementado

        Essa entidade é modelada com o modelo de 6 estados
        [x,y,vx,vy,ax,ay]
    """
    def __init__(self, dt, q_diag=None, r_diag=None, initial_P = None):
        self.dt = float(dt)

        # Matrix de transiçã ode estados já discretizada
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Matriz de observação
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=float)

        if q_diag is not None:
            Qc = np.diag(np.array(q_diag, dtype=float))
        else:
            Qc = np.diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1])

        # Aproximação generalista de Primeira Ordem para a discretização de Qc
        # Qd = Fd * (Qc * dt) * Fd^T
        self.Qd = self.F.dot(Qc * self.dt).dot(self.F.T)
        
        if r_diag is not None:
            self.R = np.diag(np.array(r_diag, dtype=float))
        else:
            self.R = np.eye(2) * 1e-1
        
        # Covariância do estado
        self.P = np.eye(6) * 500.0 if initial_P == None else initial_P
        
        # Vetor de estados
        self.x = np.zeros((6, 1), dtype=float)
        self.initialized = False

    def _g(self, x_state):
        """ Função generalizada de transição de estado g(x) do EKF """
        # Como o sistema é linear, g(x) = F * x
        return self.F.dot(x_state)

    def _jacobian_G(self, x_state):
        """ Matriz Jacobiana da função de transição (G = dg/dx) """
        # A derivada parcial de F*x em relação a x é a própria matriz constante F
        return self.F.copy()

    def _h(self, x_state):
        """ Função generalizada de medição h(x) do EKF """
        # Como a medição é afim, h(x) = H * x
        return self.H.dot(x_state)

    def _jacobian_H(self, x_state):
        """ Matriz Jacobiana da função de medição (H_jac = dh/dx) """
        # A derivada parcial de H*x em relação a x é a própria matriz constante H
        return self.H.copy()

    def initialize(self, meas):
        """
            Inicializo o estado com a primeira medição
        """
        m = np.asarray(meas).reshape(2,1)
        
        # Adicione o ", 0" para extrair o valor escalar exato (0 dimensões)
        self.x[0, 0] = float(m[0, 0])  
        self.x[1, 0] = float(m[1, 0])  
        
        self.x[2, 0] = 0.0  # vx
        self.x[3, 0] = 0.0  # vy
        self.x[4, 0] = 0.0  # ax
        self.x[5, 0] = 0.0  # ay
        self.initialized = True
        
    # Predição da posição da entidade
    def predict(self):
        # --- Passo EKF: Obtenção da matriz Jacobiana baseada no estado atual ---
        G_jac = self._jacobian_G(self.x)
        
        # Atualizando a crença a priori da entidade (belief)
        # Substituído a multiplicação direta F.dot pela função g(x) do EKF
        self.x = self._g(self.x)
        self.P = G_jac.dot(self.P).dot(G_jac.T) + self.Qd

    # Atualização da posição da entidade
    def update(self, meas):
        z = np.asarray(meas).reshape(2,1)

        # Se não tiver sido inicializado anteriormente, primeiro inicializo
        if not self.initialized:
            self.initialize(z)
            return 
            
        # --- Passo EKF: Obtenção da matriz Jacobiana de medição ---
        H_jac = self._jacobian_H(self.x)
        
        # Inovação
        # Substituído H.dot pela função não-linear h(x)
        y = z - self._h(self.x)

        # covariância da Inovação 
        # Utiliza-se a matriz Jacobiana H_jac do EKF
        S = H_jac.dot(self.P).dot(H_jac.T) + self.R

        # Ganho de Kalman (garante a minimização do erro quadrático médio)
        K = self.P.dot(H_jac.T).dot(np.linalg.inv(S))

        ## Atualização da Crença a posteriori da entidade belief()
        # Atualização do estado e da predição (u_x)
        self.x = self.x + K.dot(y)
        
        # Atualização da covariância interna do estado
        self.P = (np.eye(6) - K.dot(H_jac)).dot(self.P)


    def get_state(self):
        """ Retorno o vetor de estados completos"""
        return self.x.copy()

    
    def get_position(self):
        """ Puxo a posição [x,y]"""
        return self.x[:2].reshape(2,)

    def __str__(self):
        """Retorna uma linha de resumo rápido do estado atual do objeto."""
        if not self.initialized:
            return "[Entidy] Status: Não inicializada"
        pos = self.get_position()
        vel = self.x[2:4].reshape(2,)
        return f"[Entidy] Posição: ({pos[0]:.2f}, {pos[1]:.2f}) | Velocidade: ({vel[0]:.2f}, {vel[1]:.2f})"

    def get_uncertainty(self):
        """Retorna o desvio padrão (incerteza) atual para os eixos X e Y."""
        # A incerteza é a raiz quadrada da variância (elementos da diagonal de P)
        std_x = np.sqrt(self.P[0, 0])
        std_y = np.sqrt(self.P[1, 1])
        return std_x, std_y

    def print_summary(self, world=None):
        """
        Imprime no console um relatório completo, limpo e estruturado dos dados 
        cinemáticos e estatísticos obtidos pelo Filtro de Kalman.
        """
        print("\n" + "="*55)
        print("               RELATÓRIO DE TELEMETRIA (EKF)           ")
        print("="*55)
        
        if not self.initialized:
            print(" STATUS: Aguardando primeira medição do sensor...")
            print("="*55)
            return

        # Captura de variáveis individuais
        x, y = self.get_position()
        vx, vy = self.x[2, 0], self.x[3, 0]
        ax, ay = self.x[4, 0], self.x[5, 0]
        std_x, std_y = self.get_uncertainty()

        # Cálculos de Intensidade Resultante (Magnitude Vetorial)
        velocidade_total = np.sqrt(vx**2 + vy**2)
        aceleracao_total = np.sqrt(ax**2 + ay**2)

        print(f" STATUS: Ativo e Rastreado")
        print(f" • Posição Real:     X = {x:7.3f} m  |  Y = {y:7.3f} m")
        print(f" • Incerteza (±σ):   X = {std_x:7.3f} m  |  Y = {std_y:7.3f} m")
        print(f" • Velocidade:      Vx = {vx:7.3f} m/s | Vy = {vy:7.3f} m/s")
        print(f"   --> Vetor Velocidade Resultante: {velocidade_total:.3f} m/s")
        print(f" • Aceleração:      Ax = {ax:7.3f} m/s²| Ay = {ay:7.3f} m/s²")
        print(f"   --> Vetor Aceleração Resultante: {aceleracao_total:.3f} m/s²")

        # Integração inteligente com a classe World
        if world is not None:
            print("-"*55)
            try:
                px, py = world.world2img(x, y)
                print(f" • Mapeamento de Tela: Equivalente ao Pixel ({px}, {py})")
            except Exception as e:
                print(f" [Aviso] Não foi possível converter para pixels: {e}")
                
        print("="*55)