import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class TrajectoryGenerator:
    def __init__(self, dt: float, bases: np.ndarray):
        """
        Inicializa o gerador de trajetórias.
        dt: Intervalo de tempo entre amostras (s)
        bases: Posição das antenas base (N, 2) [[bx1, by1], ...]
        """
        self.dt = dt
        self.bases = bases
        
        # Plot styling
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig_size = (10, 8)

    def _init_vectors(self, num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inicializa os vetores de tempo e estado PVA (6D)."""
        t = np.linspace(0, (num_steps - 1) * self.dt, num_steps)
        # Estados Verdadeiros (Ground Truth): [px, py, vx, vy, ax, ay]
        states_gt = np.zeros((num_steps, 6))
        return t, states_gt

    def generate_circle(self, radius: float, center: Tuple[float, float], 
                        linear_velocity: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma trajetória circular com cinemática PVA completa.
        A aceleração gerada é a aceleração centrípeta.
        """
        num_steps = int(duration / self.dt)
        t, states = self._init_vectors(num_steps)
        cx, cy = center
        
        # Velocidade angular (w = v / r)
        omega = linear_velocity / radius
        
        # Ângulo em função do tempo (theta = w * t)
        theta = omega * t
        
        # Posições: [px, py]
        states[:, 0] = cx + radius * np.cos(theta)
        states[:, 1] = cy + radius * np.sin(theta)
        
        # Velocidades (derivada da posição): [vx, vy]
        states[:, 2] = -radius * omega * np.sin(theta)
        states[:, 3] =  radius * omega * np.cos(theta)
        
        # Acelerações (derivada da velocidade): [ax, ay]
        states[:, 4] = -radius * (omega**2) * np.cos(theta)
        states[:, 5] = -radius * (omega**2) * np.sin(theta)
        
        return t, states

    def generate_square(self, side_length: float, bottom_left: Tuple[float, float], 
                    linear_velocity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma trajetória quadrada fechada.
        O número de pontos é ajustado para que o último estado seja idêntico ao inicial.
        """
        x0, y0 = bottom_left
        v = linear_velocity
        
        # Tempo e passos por aresta
        time_per_side = side_length / v
        steps_per_side = int(time_per_side / self.dt)
        
        # num_steps + 1 para incluir o ponto de fechamento (o retorno à origem)
        num_steps = (steps_per_side * 4) + 1
        t, states = self._init_vectors(num_steps)
        
        # Aresta 1: Direita (v_x = v, v_y = 0)
        idx = slice(0, steps_per_side)
        states[idx, 0] = x0 + v * t[idx]
        states[idx, 1] = y0
        states[idx, 2] = v
        
        # Aresta 2: Cima (v_x = 0, v_y = v)
        idx = slice(steps_per_side, 2 * steps_per_side)
        states[idx, 0] = x0 + side_length
        states[idx, 1] = y0 + v * (t[idx] - time_per_side)
        states[idx, 3] = v
        
        # Aresta 3: Esquerda (v_x = -v, v_y = 0)
        idx = slice(2 * steps_per_side, 3 * steps_per_side)
        states[idx, 0] = x0 + side_length - v * (t[idx] - 2 * time_per_side)
        states[idx, 1] = y0 + side_length
        states[idx, 2] = -v
        
        # Aresta 4: Baixo (v_x = 0, v_y = -v)
        # Aqui vamos até o penúltimo ponto para não sobrescrever o fechamento manualmente
        idx = slice(3 * steps_per_side, 4 * steps_per_side)
        states[idx, 0] = x0
        states[idx, 1] = y0 + side_length - v * (t[idx] - 3 * time_per_side)
        states[idx, 3] = -v
    
        # Forçar o último ponto a ser exatamente igual ao inicial
        # Isso corrige erros de precisão de ponto flutuante e fecha o ciclo.
        states[-1, 0] = x0
        states[-1, 1] = y0
        states[-1, 2] = 0.0 # Velocidade final opcionalmente zero ou v_x da aresta 1
        states[-1, 3] = 0.0
    
        return t, states

    def generate_tanh_curve(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                            amplitude_y: float, smoothness: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma trajetória suave baseada na tangente hiperbólica (curva em S),
        semelhante a manobras de troca de faixa, com PVA completo.
        """
        num_steps = int(duration / self.dt)
        t, states = self._init_vectors(num_steps)
        x_start, y_start = start_pos
        x_end, y_end = end_pos
        
        # Tempo normalizado (0 a 1) e centralizado (-0.5 a 0.5)
        t_norm = (t / duration) - 0.5
        
        # --- Eixo X (Movimento Linear constante) ---
        dist_x = x_end - x_start
        v_x_const = dist_x / duration
        states[:, 0] = x_start + v_x_const * t # Pos X
        states[:, 2] = v_x_const                # Vel X
        # Acc X = 0
        # --- Eixo Y (Curva Tanh) ---
        y_mid = (y_end + y_start) / 2.0
        
        # Argumento da tanh: controla a inclinação da curva
        # arg = smoothness * t_norm
        
        # Posição Y: y = y_mid + Amp * tanh(smoothness * t_norm)
        tanh_arg = smoothness * t_norm
        states[:, 1] = y_mid + amplitude_y * np.tanh(tanh_arg)
        
        # Derivadas analíticas para Velocidade e Aceleração em Y
        # d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
        # Necessário usar regra da cadeia: d(smoothness*t_norm)/dt = smoothness / duration
        sech2_val = 1.0 - (np.tanh(tanh_arg))**2
        factor_dt = smoothness / duration
        
        # Velocidade Y: v_y = Amp * factor_dt * sech^2(arg)
        states[:, 3] = amplitude_y * factor_dt * sech2_val
        
        # Aceleração Y: a_y = d/dt v_y = Amp * factor_dt^2 * d/d(arg)sech^2(arg)
        # d/dx sech^2(x) = -2 * sech^2(x) * tanh(x)
        states[:, 5] = amplitude_y * (factor_dt**2) * (-2.0 * sech2_val * np.tanh(tanh_arg))
        
        return t, states
    def generate_lemniscate(self, amplitude: float, center: Tuple[float, float], 
                            linear_velocity: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma trajetória em forma de Lemniscata (Símbolo do Infinito/Curva de Lissajous).
        """
        num_steps = int(duration / self.dt)
        t, states = self._init_vectors(num_steps)
        cx, cy = center
        
        # A velocidade máxima nesta curva ocorre no centro.
        # Estimativa de omega para manter a velocidade máxima próxima da linear_velocity
        omega = linear_velocity / (amplitude * np.sqrt(5))
        
        # Equações paramétricas (x = A * sin(wt), y = A/2 * sin(2wt))
        theta = omega * t
        
        # Posições
        states[:, 0] = cx + amplitude * np.sin(theta)
        states[:, 1] = cy + (amplitude / 2.0) * np.sin(2 * theta)
        
        # Velocidades (Derivadas de X e Y)
        states[:, 2] = amplitude * omega * np.cos(theta)
        states[:, 3] = amplitude * omega * np.cos(2 * theta)
        
        # Acelerações (Derivadas de Vx e Vy)
        states[:, 4] = -amplitude * (omega**2) * np.sin(theta)
        states[:, 5] = -2 * amplitude * (omega**2) * np.sin(2 * theta)
        
        return t, states

    def generate_random(self, start_pos: Tuple[float, float], initial_velocity: float, 
                        noise_std: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma trajetória com caminhada aleatória suave (Random Walk nas Acelerações).
        """
        num_steps = int(duration / self.dt)
        t, states = self._init_vectors(num_steps)
        
        # Início
        states[0, 0], states[0, 1] = start_pos
        
        # Direção inicial aleatória
        angle = np.random.uniform(0, 2 * np.pi)
        states[0, 2] = initial_velocity * np.cos(angle)
        states[0, 3] = initial_velocity * np.sin(angle)
        
        # Iteração de Integração Euler Simples para gerar a trajetória
        for i in range(1, num_steps):
            # Aceleração recebe um ruído Gaussiano
            ax = np.random.normal(0, noise_std)
            ay = np.random.normal(0, noise_std)
            
            # Fator de amortecimento leve para a velocidade não explodir infinitamente
            damping = 0.98 
            
            states[i, 4] = ax
            states[i, 5] = ay
            
            # Atualiza velocidade: v = v0*damp + a*dt
            states[i, 2] = states[i-1, 2] * damping + ax * self.dt
            states[i, 3] = states[i-1, 3] * damping + ay * self.dt
            
            # Atualiza posição: p = p0 + v*dt + 0.5*a*dt^2
            states[i, 0] = states[i-1, 0] + states[i-1, 2] * self.dt + 0.5 * ax * (self.dt**2)
            states[i, 1] = states[i-1, 1] + states[i-1, 3] * self.dt + 0.5 * ay * (self.dt**2)
            
        return t, states

    def generate_occlusion(self, radius: float, center: Tuple[float, float], 
                           linear_velocity: float, duration: float, 
                           occlusion_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera uma trajetória circular e retorna também uma máscara de visibilidade (booleana).
        True = Medição disponível, False = Objeto Oculto (Sumiço).
        """
        # Utiliza a cinemática do círculo como Ground Truth
        t, states = self.generate_circle(radius, center, linear_velocity, duration)
        
        # Cria a máscara (todos iniciam visíveis)
        visible_mask = np.ones(len(t), dtype=bool)
        
        # Define a oclusão no meio da trajetória
        mid_idx = len(t) // 2
        half_occ = occlusion_frames // 2
        
        start_occ = max(0, mid_idx - half_occ)
        end_occ = min(len(t), mid_idx + half_occ)
        
        # Marca o trecho como oculto (False)
        visible_mask[start_occ:end_occ] = False
        
        return t, states, visible_mask
    
    def plot_scenario(self, t: np.ndarray, states_gt: np.ndarray, title: str):
        """Planta a trajetória (Ground Truth) e as bases."""
        px = states_gt[:, 0]
        py = states_gt[:, 1]
        vx = states_gt[:, 2]
        vy = states_gt[:, 3]
        ax = states_gt[:, 4]
        ay = states_gt[:, 5]
        
        # Cria figura com subplots (Trajetória + Cinematica)
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2)
        
        # --- Plot 1: Espacial (X vs Y) ---
        ax_space = fig.add_subplot(gs[0:2, 0])
        
        # Plot Trajetória ideal
        ax_space.plot(px, py, 'g-', linewidth=2.5, label='Trajetória Ideal (GT)')
        
        # Início e Fim
        ax_space.plot(px[0], py[0], 'go', markersize=8, label='Início')
        ax_space.plot(px[-1], py[-1], 'gx', markersize=10, mew=2, label='Fim')
        
        # Plot Bases
        ax_space.scatter(self.bases[:, 0], self.bases[:, 1], 
                         marker='^', s=150, c='red', edgecolor='black', 
                         label='Antenas Base', zorder=5)
        # Numeração das bases
        for i, (bx, by) in enumerate(self.bases):
            ax_space.annotate(f'B{i}', (bx, by), xytext=(5, 5), textcoords='offset points', fontsize=16, fontweight='bold')
            
        ax_space.set_title(f'Cenário Espacial: {title}', fontsize=14)
        ax_space.set_xlabel('Posição X (m)', fontsize=16)
        ax_space.set_ylabel('Posição Y (m)', fontsize=16)
        ax_space.legend(loc='best', frameon=True, shadow=True)
        ax_space.axis('equal') # Mantém proporção 1:1 m
        ax_space.grid(True, which='both', linestyle='--')

        # --- Subplots Cinemática (Tempo) ---
        # Velocidade
        ax_vel = fig.add_subplot(gs[0, 1])
        ax_vel.plot(t, vx, 'b--', label='$v_x$')
        ax_vel.plot(t, vy, 'r--', label='$v_y$')
        ax_vel.set_title('Velocidade Ideal', fontsize=16)
        ax_vel.set_ylabel('(m/s)')
        ax_vel.legend()
        ax_vel.grid(True)
        
        # Aceleração
        ax_acc = fig.add_subplot(gs[1, 1])
        ax_acc.plot(t, ax, 'b-', label='$a_x$')
        ax_acc.plot(t, ay, 'r-', label='$a_y$')
        ax_acc.set_title('Aceleração Ideal (Necessária para a curva)', fontsize=16)
        ax_acc.set_ylabel('(m/$s^2$)')
        ax_acc.legend()
        ax_acc.grid(True)
        
        # Tanh Y específica (para mostrar a forma)
        ax_tanh = fig.add_subplot(gs[2, 1])
        ax_tanh.plot(t, py, 'k-', label='$p_y$ (Forma Tanh)')
        ax_tanh.set_title('Perfil de Posição Y (Forma da Tanh)', fontsize=16)
        ax_tanh.set_xlabel('Tempo (s)')
        ax_tanh.set_ylabel('(m)')
        ax_tanh.grid(True)
        
        fig.tight_layout()
        plt.show()
