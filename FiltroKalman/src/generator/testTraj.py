from trajectories import *

if __name__ == "__main__":
    # 1. Configurações Iniciais
    dt = 0.1
    antenas_base = np.array([
        [-45, -45],
        [45, -45],
        [45, 45],
        [-45, 45]
    ])
    
    generator = TrajectoryGenerator(dt=dt, bases=antenas_base)
    
    # 2. Geração dos Dados
    _, state_circle = generator.generate_circle(radius=25, center=(0, 0), linear_velocity=10.0, duration=40.0)
    _, state_square = generator.generate_square(side_length=50, bottom_left=(-25, -25), linear_velocity=10.0)
    _, state_tanh = generator.generate_tanh_curve(start_pos=(-42, -20), end_pos=(42, 20), amplitude_y=20, smoothness=5.0, duration=30.0)
    _, state_lemniscate = generator.generate_lemniscate(amplitude=35, center=(0, 0), linear_velocity=12.0, duration=50.0)
    _, state_random = generator.generate_random(start_pos=(0, 0), initial_velocity=4.0, noise_std=1.0, duration=60.0)
    _, state_occlusion, visible_mask = generator.generate_occlusion(start_pos=(0, 0), initial_velocity=4.0, noise_std=1.0, duration=60.0, occlusion_frames=25)
    
    # 3. Configuração do Grid de Subplots (Agora 2x3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    # Cálculo dinâmico dos limites (Região das torres + margem)
    margem = 5
    x_min, y_min = np.min(antenas_base, axis=0) - margem
    x_max, y_max = np.max(antenas_base, axis=0) + margem
    
    # Lista para plotagem em loop das 5 primeiras
    trajetorias = [
        ("1. Circular", state_circle, 'g-'),
        ("2. Quadrada", state_square, 'b-'),
        ("3. Curva Tanh", state_tanh, 'm-'),
        ("4. Lemniscata", state_lemniscate, 'c-'),
        ("5. Aleatória", state_random, 'orange')
    ]
    
    # Plot das 5 primeiras trajetórias
    for idx, (title, state, style) in enumerate(trajetorias):
        ax = axes[idx]
        ax.plot(state[:, 0], state[:, 1], style, linewidth=1.5, label='Trajetória Ideal')
        ax.plot(state[0, 0], state[0, 1], 'go', markersize=5, label='Início')
        ax.plot(state[-1, 0], state[-1, 1], 'kx', markersize=6, mew=1.5, label='Fim')
        ax.set_title(title, fontsize=11, fontweight='bold')
        
    # Tratamento da 6ª Trajetória (Oclusões)
    ax_occ = axes[5]
    ax_occ.set_title("6. Aleatória com Oclusões", fontsize=11, fontweight='bold')
    ax_occ.plot(state_occlusion[:, 0], state_occlusion[:, 1], 'k--', alpha=0.3, linewidth=1.2, label='Trecho Oculto')
    
    p_visivel = state_occlusion[visible_mask]
    ax_occ.scatter(p_visivel[:, 0], p_visivel[:, 1], c='green', s=4, zorder=3, label='Sinal Visível')
    ax_occ.plot(state_occlusion[0, 0], state_occlusion[0, 1], 'go', markersize=5)
    ax_occ.plot(state_occlusion[-1, 0], state_occlusion[-1, 1], 'kx', markersize=6, mew=1.5)

    # 4. Formatação Estética Global e Torres
    for ax in axes:
        # Torres discretas nas quinas
        ax.scatter(antenas_base[:, 0], antenas_base[:, 1], 
                   marker='^', s=60, c='red', edgecolor='black', linewidth=0.8,
                   label='Antenas Base', zorder=5)
        
        # Identificadores das torres
        for k, (bx, by) in enumerate(antenas_base):
            ax.annotate(f'B{k}', (bx, by), xytext=(4, 4), textcoords='offset points', 
                        fontsize=8, color='darkred', fontweight='bold')
            
        # Formatação sutil dos eixos
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        
        # Trava o zoom estritamente na região das torres
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
    # 5. Construção Automática da Legenda Global (Sem duplicatas)
    handles_unicos = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_unicos:
                handles_unicos[label] = handle
                
    # Cria a legenda única posicionada centralizada na parte inferior
    fig.legend(handles_unicos.values(), handles_unicos.keys(), 
               loc='lower center', ncol=6, fontsize=9, frameon=True, shadow=False)
               
    # Ajustes finos: hspace adiciona o respiro ideal entre as linhas do grid
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.96, hspace=0.35)
    
    plt.show()