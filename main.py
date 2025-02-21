import pygame
import numpy as np
import random
import time

# Parâmetros do Q-Learning
alpha = 1     # Taxa de aprendizado (0 - 1)
gamma = 0.9     # Fator de desconto
epsilon = 0.2   # Taxa de exploração inicial
min_epsilon = 0.01  # Valor mínimo de epsilon
decay_rate = 0.995   # Taxa de decaimento de epsilon
actions = [
    (0, -1),    # cima
    (0, 1),     # baixo
    (-1, 0),    # esquerda
    (1, 0)      # direita
]

class QLearnAgent:
    def __init__(self, maze_size):
        # Inicializa a tabela Q com zeros (estados x ações)
        self.q_table = {}
        self.maze_size = maze_size
        self.current_epsilon = epsilon

    def get_q_value(self, state, action_idx):
        # Retorna o valor Q atual para o par estado-ação
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(actions))
        return self.q_table[state][action_idx]

    def update_q_value(self, state, action_idx, reward, next_state):
        # Implementa a fórmula de atualização do Q-Learning
        max_next_q = max([self.get_q_value(next_state, a) for a in range(len(actions))])
        current_q = self.get_q_value(state, action_idx)

        # Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

        # Atualiza a tabela Q
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(actions))
        self.q_table[state][action_idx] = new_q

    def choose_action(self, state, exploit_only=False):
        # No modo de exploração zero, sempre escolhe a melhor ação
        if exploit_only or random.random() > self.current_epsilon:
            # Aproveitamento: escolhe a melhor ação conhecida
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(actions))
            return np.argmax(self.q_table[state])
        else:
            # Exploração: escolhe uma ação aleatória
            return random.randint(0, len(actions) - 1)

    # Obter o melhor caminho baseado na política aprendida
    def get_best_path(self, start_pos, goal_pos, is_valid_move_func, max_steps=1000):
        current = start_pos
        path = [current]
        steps = 0

        while current != goal_pos and steps < max_steps:
            action_idx = self.choose_action(current, exploit_only=True)
            dx, dy = actions[action_idx]
            new_pos = (current[0] + dx, current[1] + dy)

            if is_valid_move_func(new_pos):
                current = new_pos
                path.append(current)

            steps += 1

            # Evitar loops infinitos
            if steps >= max_steps:
                break

        return path

    def decay_epsilon(self):
        # Diminui a taxa de exploração ao longo do tempo
        self.current_epsilon = max(min_epsilon, self.current_epsilon * decay_rate)

def main():
    pygame.init()
    clock = pygame.time.Clock()

    # Carrega imagens
    maze_image = pygame.image.load("labirinto.png")
    original_size = maze_image.get_size()  # Tamanho original (32x32)
    scaled_size = (640, 640)               # Tamanho escalado para visualização

    maze_surface = pygame.transform.scale(maze_image, scaled_size)
    ia_image = pygame.image.load("ia.png")
    ia_surface = pygame.transform.scale(ia_image, (20, 20))  # Aumentei um pouco para melhor visibilidade

    screen = pygame.display.set_mode(scaled_size)
    pygame.display.set_caption("IA no Labirinto - Q-Learning")

    # Analisa o labirinto para encontrar entrada, saída e criar mapa de colisão
    maze_data = pygame.surfarray.array3d(maze_image)

    # Cores para identificação: Amarelo (entrada), Vermelho (saída), Branco (caminho), Preto (parede)
    start_pos = None
    goal_pos = None
    collision_map = np.zeros((original_size[0], original_size[1]), dtype=bool)

    # Escaneia o labirinto pixel por pixel
    for y in range(original_size[1]):
        for x in range(original_size[0]):
            pixel = maze_data[x][y]

            # Detecta entrada (amarelo)
            if pixel[0] > 200 and pixel[1] > 200 and pixel[2] < 100:
                start_pos = (x, y)

            # Detecta saída (vermelho)
            elif pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 100:
                goal_pos = (x, y)

            # Detecta parede (preto)
            elif pixel[0] < 50 and pixel[1] < 50 and pixel[2] < 50:
                collision_map[x][y] = True

    if start_pos is None or goal_pos is None:
        print("Erro: Não foi possível encontrar a entrada ou saída no labirinto!")
        pygame.quit()
        return

    print(f"Posição inicial: {start_pos}, Posição final: {goal_pos}")

    # Inicializa o agente Q-Learning
    agent = QLearnAgent(original_size)

    # Variáveis de controle da simulação
    current_pos = start_pos
    scale_factor = scaled_size[0] / original_size[0]
    episode = 1
    total_steps = 0
    max_steps_per_episode = 1000
    training_episodes = 500  # Número de episódios para treinamento
    demonstration_episodes = 10  # Número de episódios para demonstração

    # Rastreamento do melhor caminho
    best_path_length = float('inf')
    current_path = []
    best_path = []

    # Modo de simulação
    TRAINING_MODE = 0
    DEMO_MODE = 1
    BEST_PATH_MODE = 2
    current_mode = TRAINING_MODE

    # Função para converter coordenadas do mapa original para a visualização escalada
    def scale_pos(pos):
        x, y = pos
        return (int(x * scale_factor), int(y * scale_factor))

    # Função para verificar se um movimento é válido
    def is_valid_move(pos):
        x, y = pos
        if x < 0 or y < 0 or x >= original_size[0] or y >= original_size[1]:
            return False
        return not collision_map[x][y]

    # Função para calcular a recompensa
    def get_reward(pos, steps_taken):
        if pos == goal_pos:
            # Recompensa maior para caminhos mais curtos
            return 100 + (1000 / max(1, steps_taken))
        elif not is_valid_move(pos):
            return -10  # Punição para colisão
        else:
            # Pequena punição por cada passo para incentivar caminhos curtos
            return -0.1

    # Função para desenhar o rastro da IA
    def draw_path(surface, path, color=(0, 255, 0, 128)):
        path_surface = pygame.Surface(scaled_size, pygame.SRCALPHA)
        for pos in path:
            x, y = scale_pos(pos)
            pygame.draw.circle(path_surface, color,
                              (x + scale_factor/2, y + scale_factor/2), 5)
        surface.blit(path_surface, (0, 0))

    print("== Iniciando treinamento da IA ==")
    running = True
    training_speed = 1000  # FPS para treinamento
    demo_speed = 10      # FPS para demonstração
    best_path_speed = 5  # FPS para demonstração do melhor caminho
    steps_in_episode = 0
    show_best_path_preview = False
    best_path_simulation_pos = 0

    font = pygame.font.SysFont(None, 24)

    while running:
        # Processa eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Alternar entre velocidade rápida e lenta
                    if current_mode == TRAINING_MODE:
                        training_speed = 60 if training_speed == 10 else 10
                    elif current_mode == DEMO_MODE:
                        demo_speed = 10 if demo_speed == 2 else 2
                    else:  # BEST_PATH_MODE
                        best_path_speed = 10 if best_path_speed == 2 else 2

                elif event.key == pygame.K_v:
                    # Ativar/desativar visualização do melhor caminho
                    if current_mode != BEST_PATH_MODE:
                        # Calcular o melhor caminho baseado no conhecimento atual
                        calculated_path = agent.get_best_path(start_pos, goal_pos, is_valid_move)
                        if len(calculated_path) > 1 and calculated_path[-1] == goal_pos:
                            print(f"Visualizando melhor caminho atual ({len(calculated_path)} passos)")
                            best_path = calculated_path
                            previous_mode = current_mode
                            current_mode = BEST_PATH_MODE
                            best_path_simulation_pos = 0
                        else:
                            print("Ainda não foi encontrado um caminho válido")
                    else:
                        # Voltar ao modo anterior
                        current_mode = previous_mode
                        current_pos = start_pos
                        current_path = []
                        steps_in_episode = 0

                elif event.key == pygame.K_r:
                    # Reiniciar episódio
                    current_pos = start_pos
                    current_path = []
                    steps_in_episode = 0

        # Lógica baseada no modo atual
        if current_mode == BEST_PATH_MODE:
            # Animação do melhor caminho
            if best_path_simulation_pos < len(best_path):
                current_pos = best_path[best_path_simulation_pos]
                best_path_simulation_pos += 1

                # Chegou ao final do caminho, pausa e depois reinicia
                if best_path_simulation_pos >= len(best_path):
                    time.sleep(1)
                    best_path_simulation_pos = 0

        else:  # Modo TRAINING ou DEMO
            # Estado atual
            state = current_pos

            # Rastreia caminho atual
            if current_pos not in current_path:
                current_path.append(current_pos)

            # Escolhe uma ação baseada no modo atual
            if current_mode == TRAINING_MODE:
                action_idx = agent.choose_action(state)
            else:  # Modo de demonstração - sempre usa a melhor ação
                action_idx = agent.choose_action(state, exploit_only=True)

            dx, dy = actions[action_idx]

            # Calcula nova posição
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)

            # Verifica se está dentro dos limites e não é parede
            if is_valid_move(new_pos):
                next_pos = new_pos
            else:
                next_pos = current_pos  # Permanece na mesma posição se colidir

            # Calcula recompensa
            reward = get_reward(next_pos, steps_in_episode)

            # Atualiza Q-table apenas no modo de treinamento
            if current_mode == TRAINING_MODE:
                agent.update_q_value(state, action_idx, reward, next_pos)

            # Atualiza posição atual
            current_pos = next_pos
            steps_in_episode += 1
            total_steps += 1

        # Renderiza
        screen.blit(maze_surface, (0, 0))

        # Desenha o caminho atual
        if current_mode == DEMO_MODE:
            draw_path(screen, current_path, (100, 100, 255, 100))

        # Desenha o melhor caminho
        if current_mode == BEST_PATH_MODE:
            # Desenha o caminho completo
            draw_path(screen, best_path[:best_path_simulation_pos], (0, 255, 0, 200))
            # Desenha a parte que ainda vai ser percorrida com outra cor
            draw_path(screen, best_path[best_path_simulation_pos:], (0, 200, 0, 100))

        # Desenha a IA na posição atual, centralizada
        ia_pos = scale_pos(current_pos)
        ia_rect = ia_surface.get_rect(center=(
            ia_pos[0] + scale_factor/2,
            ia_pos[1] + scale_factor/2
        ))
        screen.blit(ia_surface, ia_rect)

        # Desenha informações na tela
        if current_mode == TRAINING_MODE:
            mode_text = "TREINAMENTO"
        elif current_mode == DEMO_MODE:
            mode_text = "DEMONSTRAÇÃO"
        else:
            mode_text = "VISUALIZAÇÃO MELHOR CAMINHO"

        info_text = [
            f"Modo: {mode_text}",
            f"Episódio: {episode}",
            f"Passos no episódio: {steps_in_episode}",
            f"Total de passos: {total_steps}",
            f"Epsilon: {agent.current_epsilon:.3f}",
            f"Melhor caminho: {len(best_path)} passos",
            "ESPAÇO: Alternar velocidade",
            "V: Visualizar melhor caminho",
            "R: Reiniciar episódio",
        ]

        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (210, 100, 100))
            screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()

        # Controla velocidade baseada no modo
        if current_mode == TRAINING_MODE:
            clock.tick(training_speed)
        elif current_mode == DEMO_MODE:
            clock.tick(demo_speed)
        else:  # BEST_PATH_MODE
            clock.tick(best_path_speed)

        # Verifica condições de término do episódio
        if current_pos == goal_pos and current_mode != BEST_PATH_MODE:
            # Registra o melhor caminho se for mais curto
            path_length = len(set(current_path))  # Remove posições duplicadas

            if path_length < best_path_length and current_mode == TRAINING_MODE:
                best_path_length = path_length
                best_path = current_path.copy()
                print(f"Novo melhor caminho encontrado! Comprimento: {best_path_length}")

            print(f"Episódio {episode} concluído! Saída encontrada em {steps_in_episode} passos.")

            # Avança para o próximo episódio
            episode += 1
            current_pos = start_pos
            current_path = []
            steps_in_episode = 0

            # Reduz epsilon para favorecer a exploração de conhecimento
            if current_mode == TRAINING_MODE:
                agent.decay_epsilon()

            # Pausa breve para ver o sucesso
            time.sleep(0.5)

            # Muda para modo de demonstração após terminar treinamento
            if current_mode == TRAINING_MODE and episode > training_episodes:
                current_mode = DEMO_MODE
                episode = 1
                print("\n== Treinamento concluído! Iniciando demonstração ==")
                print(f"Melhor caminho encontrado: {best_path_length} passos")

            # Termina após os episódios de demonstração
            if current_mode == DEMO_MODE and episode > demonstration_episodes:
                # Alterna para modo de melhor caminho automaticamente
                current_mode = BEST_PATH_MODE
                best_path_simulation_pos = 0
                episode = 1
                print("\n== Demonstração concluída! Mostrando melhor caminho ==")

        elif steps_in_episode >= max_steps_per_episode and current_mode != BEST_PATH_MODE:
            print(f"Episódio {episode} - Limite de passos atingido.")
            episode += 1
            current_pos = start_pos
            current_path = []
            steps_in_episode = 0

            # Muda para modo de demonstração após terminar treinamento
            if current_mode == TRAINING_MODE and episode > training_episodes:
                current_mode = DEMO_MODE
                episode = 1
                print("\n== Treinamento concluído! Iniciando demonstração ==")

    print(f"Simulação concluída após {episode-1} episódios e {total_steps} passos totais.")
    print(f"Melhor caminho encontrado: {best_path_length} passos")
    pygame.quit()

if __name__ == "__main__":
    main()
