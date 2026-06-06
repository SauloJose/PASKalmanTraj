"""
Custom object detector for Kalman Filter tracking.

Este módulo contém o código de detecção de objetos que será utilizado
pelo sistema de Kalman para gerar as medições de posição.
"""

import cv2
import numpy as np

def detect_centroid(frame, noise_std=3.0):
    """
    Detecta o centróide de um objeto laranja no frame usando HSV.
    Permite adicionar ruído gaussiano para simular um sensor imperfeito.

    Args:
        frame: numpy array BGR (formato OpenCV)
        noise_std: Desvio padrão do ruído gaussiano a ser adicionado (em pixels).
                   0.0 significa medição perfeita.

    Returns:
        Tupla (cX, cY) com coordenadas do centróide (com ruído se aplicável).
        None se nenhum objeto for detectado.
    """ 
    # Converte a imagem de BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites da cor LARANJA no espaço HSV
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([25, 255, 255])

    # Limiarização: cria uma máscara apenas com os pixels que estão nesse limiar
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Operações Morfológicas
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Encontra os contornos na máscara resultante
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None 
    
    # Pegar o maior contorno
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Se a área for muito pequena, considero como um falso positivo
    if area < 10:
        return None 
    
    # Calcula o centróide real do objeto usando Momentos da Imagem
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None 
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # PARTE IMPORTANTE => ADIÇÃO DE RUÍDOS PARA SIMULAR UM SENSOR REAL
    if noise_std > 0:
        ruido_x = np.random.normal(0, noise_std)
        ruido_y = np.random.normal(0, noise_std)

        # Aplica o ruído e garante que as coordenadas finais continuem a ser números inteiros
        cX = int(cX + ruido_x)
        cY = int(cY + ruido_y)
    
    return (cX, cY)

def detect_color_orange(frame, noise_std=0.0):
    return detect_centroid(frame, noise_std)