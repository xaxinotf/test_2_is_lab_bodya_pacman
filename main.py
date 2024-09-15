import pygame
import sys
import random
import heapq

# Ініціалізація Pygame
pygame.init()

# Параметри екрану
TILE_SIZE = 20
MAZE_WIDTH = 27  # Непарне число для генерації лабіринту
MAZE_HEIGHT = 31
screen_width = MAZE_WIDTH * TILE_SIZE
screen_height = MAZE_HEIGHT * TILE_SIZE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pac-Man Cheban Bogdan TTP-42_lab-1")

clock = pygame.time.Clock()

# Кольори
BLACK = (0, 0, 0)
BLUE = (0, 0, 155)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 105, 180)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)

# Напрямки
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Класи
class Wall(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(topleft=pos)

class Dot(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((5, 5))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(center=(pos[0]+TILE_SIZE//2, pos[1]+TILE_SIZE//2))

class PowerPellet(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(center=(pos[0]+TILE_SIZE//2, pos[1]+TILE_SIZE//2))

class PacManAI(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self.original_image = self.image.copy()
        pygame.draw.circle(self.image, YELLOW, (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
        self.rect = self.image.get_rect(center=pos)
        self.speed = 1.5
        self.lives = 3
        self.score = 0
        self.direction = (0, 0)
        self.path = []
        self.maze = None
        self.powered_up = False
        self.powerup_timer = 0
        self.path_timer = 0  # Таймер для перерахунку шляху

    def update(self, walls):
        if self.maze is None:
            return

        self.path_timer += 1
        if self.path_timer >= 15 or not self.path:
            self.path_timer = 0
            self.calculate_new_path()

        # Рух по шляху
        if self.path and len(self.path) > 0:
            next_cell = self.path[0]
            target_x = next_cell[0] * TILE_SIZE + TILE_SIZE // 2
            target_y = next_cell[1] * TILE_SIZE + TILE_SIZE // 2
            dx = target_x - self.rect.centerx
            dy = target_y - self.rect.centery
            dist = (dx**2 + dy**2)**0.5
            if dist != 0:
                move_x = self.speed * dx / dist
                move_y = self.speed * dy / dist
                self.rect.centerx += move_x
                self.rect.centery += move_y

                # Перевірка на досягнення наступної клітинки
                if abs(dx) < self.speed and abs(dy) < self.speed:
                    self.rect.centerx = target_x
                    self.rect.centery = target_y
                    self.path.pop(0)
                    self.update_direction()
        else:
            # Якщо шлях не знайдено, залишаємося на місці
            pass

        # Перевірка стану power-up
        if self.powered_up:
            self.powerup_timer -= 1
            if self.powerup_timer <= 0:
                self.powered_up = False
                self.image = self.original_image.copy()
                pygame.draw.circle(self.image, YELLOW, (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)

    def update_direction(self):
        if self.path and len(self.path) > 0:
            current_pos = (int(self.rect.centerx) // TILE_SIZE, int(self.rect.centery) // TILE_SIZE)
            next_pos = self.path[0]
            self.direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        else:
            self.direction = (0, 0)

    def get_direction(self):
        return self.direction

    def get_grid_position(self):
        return (int(self.rect.centerx) // TILE_SIZE, int(self.rect.centery) // TILE_SIZE)

    def calculate_new_path(self):
        start = self.get_grid_position()
        # Знаходимо всі точки та Power Pellets
        dot_positions = [(dot.rect.centerx // TILE_SIZE, dot.rect.centery // TILE_SIZE) for dot in dots]
        pellet_positions = [(pellet.rect.centerx // TILE_SIZE, pellet.rect.centery // TILE_SIZE) for pellet in power_pellets]
        targets = dot_positions + pellet_positions
        if not targets:
            return
        # Вибираємо найкращу ціль з урахуванням ризиків
        min_total_cost = float('inf')
        best_path = []
        for pos in targets:
            path = astar(self.maze, start, pos, self.avoid_ghosts_penalty)
            if path:
                total_cost = len(path) + self.estimate_risk(path)
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_path = path
        self.path = best_path
        self.update_direction()

    def estimate_risk(self, path):
        risk = 0
        for pos in path:
            for ghost in ghosts:
                ghost_pos = (int(ghost.rect.centerx) // TILE_SIZE, int(ghost.rect.centery) // TILE_SIZE)
                distance = heuristic(pos, ghost_pos)
                if distance == 0:
                    risk += 1000
                else:
                    risk += max(0, 50 - distance * 5)
        return risk

    def avoid_ghosts_penalty(self, pos):
        penalty = 0
        for ghost in ghosts:
            ghost_pos = (int(ghost.rect.centerx) // TILE_SIZE, int(ghost.rect.centery) // TILE_SIZE)
            distance = heuristic(pos, ghost_pos)
            if distance == 0:
                penalty += 1000
            else:
                penalty += max(0, 50 - distance * 5)
        return penalty

    def notify_maze_changed(self):
        self.path = []
        self.path_timer = 0  # Додано, щоб негайно перерахувати шлях

class GameTimer:
    def __init__(self):
        self.timer = 0
        self.phase = 'chase'  # Початкова фаза
        self.phase_durations = {
            'chase': 20 * 60,      # 20 секунд переслідування
            'scatter': 7 * 60      # 7 секунд розсіювання
        }

    def update(self):
        self.timer += 1
        current_duration = self.phase_durations[self.phase]
        if self.timer >= current_duration:
            self.timer = 0
            self.switch_phase()

    def switch_phase(self):
        if self.phase == 'chase':
            self.phase = 'scatter'
        else:
            self.phase = 'chase'
        print(f"Фаза змінилася на {self.phase}")

game_timer = GameTimer()

class Ghost(pygame.sprite.Sprite):
    def __init__(self, pos, color, algorithm, role):
        super().__init__()
        self.base_image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(self.base_image, color, (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=pos)
        self.initial_position = pos
        self.speed = 1  # Зменшена швидкість для плавного руху
        self.algorithm = algorithm  # 'dfs', 'bfs', 'astar', 'heuristic'
        self.path = []
        self.maze = None
        self.path_timer = 0
        self.needs_new_path = True
        self.frightened_mode = False
        self.target = None  # Ціль для руху в frightened_mode
        self.respawn_timer = 0
        self.state = 'chase'  # Можливі стани: 'chase', 'frightened', 'respawn'
        self.role = role
        self.set_scatter_target()

    def set_scatter_target(self):
        if self.role == 'chaser':
            self.scatter_target = (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)  # Правий нижній кут
        elif self.role == 'ambusher':
            self.scatter_target = (1, 1)  # Лівий верхній кут
        elif self.role == 'patroller':
            self.scatter_target = (MAZE_WIDTH - 2, 1)  # Правий верхній кут
        elif self.role == 'random':
            self.scatter_target = (1, MAZE_HEIGHT - 2)  # Лівий нижній кут
        else:
            self.scatter_target = (MAZE_WIDTH // 2, MAZE_HEIGHT // 2)  # Центр

    def update(self, walls):
        if self.maze is None:
            return

        if self.state == 'respawn':
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                self.state = 'chase'
                self.image = self.base_image.copy()
                self.rect.center = self.initial_position
                self.path = []
                self.needs_new_path = True
            return

        if self.state == 'frightened':
            self.path_timer += 1
            if self.path_timer >= 30 or not self.path:
                self.path_timer = 0
                self.calculate_frightened_path()
            self.follow_path()
            return

        self.path_timer += 1
        if self.path_timer >= 30 or self.needs_new_path:
            self.path_timer = 0
            self.calculate_new_path()

        self.follow_path()

    def follow_path(self):
        # Рухаємось по шляху
        if self.path and len(self.path) > 0:
            next_cell = self.path[0]
            target_x = next_cell[0] * TILE_SIZE + TILE_SIZE // 2
            target_y = next_cell[1] * TILE_SIZE + TILE_SIZE // 2
            dx = target_x - self.rect.centerx
            dy = target_y - self.rect.centery
            dist = (dx**2 + dy**2)**0.5
            if dist != 0:
                move_x = self.speed * dx / dist
                move_y = self.speed * dy / dist
                self.rect.centerx += move_x
                self.rect.centery += move_y

                # Перевірка на досягнення наступної клітинки
                if abs(dx) < self.speed and abs(dy) < self.speed:
                    self.rect.centerx = target_x
                    self.rect.centery = target_y
                    self.path.pop(0)
        else:
            # Якщо шлях не знайдено або закінчився, не рухаємось
            pass

    def calculate_new_path(self):
        start = (int(self.rect.centerx) // TILE_SIZE, int(self.rect.centery) // TILE_SIZE)
        if self.maze[start[1]][start[0]] == 'X':
            return

        if game_timer.phase == 'chase':
            if self.role == 'chaser':
                goal = pacman.get_grid_position()
            elif self.role == 'ambusher':
                pac_direction = pacman.get_direction()
                goal = (pacman.get_grid_position()[0] + pac_direction[0]*4, pacman.get_grid_position()[1] + pac_direction[1]*4)
                if not self.is_valid_position(goal):
                    goal = pacman.get_grid_position()
            else:
                goal = pacman.get_grid_position()
        elif game_timer.phase == 'scatter':
            goal = self.scatter_target

        self.path = astar(self.maze, start, goal)
        self.needs_new_path = False

    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT and self.maze[y][x] != 'X'

    def calculate_frightened_path(self):
        # Вибираємо випадкову точку в лабіринті як ціль
        start = (int(self.rect.centerx) // TILE_SIZE, int(self.rect.centery) // TILE_SIZE)

        # Якщо немає цілі або ціль досягнута, обираємо нову
        if not self.target or start == self.target:
            self.target = self.get_random_target()
        if self.maze[start[1]][start[0]] == 'X' or self.maze[self.target[1]][self.target[0]] == 'X':
            return
        self.path = astar(self.maze, start, self.target)

    def get_random_target(self):
        # Отримуємо всі доступні клітинки
        available_cells = [(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH) if self.maze[y][x] == '.']
        return random.choice(available_cells)

    def enter_frightened_mode(self):
        self.state = 'frightened'
        self.speed = 0.5  # Зменшуємо швидкість
        self.image = self.base_image.copy()
        pygame.draw.circle(self.image, BLUE, (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
        self.path = []
        self.target = self.get_random_target()
        self.needs_new_path = False
        self.path_timer = 0

    def exit_frightened_mode(self):
        self.state = 'chase'
        self.speed = 1
        self.image = self.base_image.copy()
        self.path = []
        self.needs_new_path = True
        self.target = None

    def start_respawn(self):
        self.state = 'respawn'
        self.respawn_timer = 180  # 3 секунди при 60 FPS
        self.rect.center = (MAZE_WIDTH // 2 * TILE_SIZE + TILE_SIZE // 2, MAZE_HEIGHT // 2 * TILE_SIZE + TILE_SIZE // 2)
        self.image = self.base_image.copy()
        pygame.draw.circle(self.image, (128, 128, 128), (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
        self.path = []
        self.needs_new_path = False

# Пошукові алгоритми
def dfs(maze, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex == goal:
            return path[1:]  # Повертаємо шлях без початкової позиції
        if vertex not in visited:
            visited.add(vertex)
            neighbors = get_neighbors(maze, vertex)
            random.shuffle(neighbors)  # Додаємо випадковість
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return []

def bfs(maze, start, goal):
    queue = [(start, [start])]
    visited = set()
    visited.add(start)
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex == goal:
            return path[1:]
        for neighbor in get_neighbors(maze, vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

def astar(maze, start, goal, penalty_func=lambda pos: 0):
    heap = []
    heapq.heappush(heap, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    while heap:
        (est_total, cost_so_far, vertex, path) = heapq.heappop(heap)
        if vertex == goal:
            return path[1:]
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in get_neighbors(maze, vertex):
                if neighbor not in visited:
                    total_cost = cost_so_far + 1 + penalty_func(neighbor)
                    est_total_cost = total_cost + heuristic(neighbor, goal)
                    heapq.heappush(heap, (est_total_cost, total_cost, neighbor, path + [neighbor]))
    return []

def heuristic(a, b):
    # Манхеттенська відстань
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(maze, pos):
    neighbors = []
    for direction in DIRECTIONS:
        x = pos[0] + direction[0]
        y = pos[1] + direction[1]
        if 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT:
            if maze[y][x] != 'X':
                neighbors.append((x, y))
    return neighbors

# Генерація лабіринту з використанням алгоритму Прима
def generate_maze(level):
    maze = [['X'] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
    walls_list = []
    start_x = MAZE_WIDTH // 2
    start_y = MAZE_HEIGHT // 2
    maze[start_y][start_x] = '.'

    walls_list.extend([(start_x + dx, start_y + dy, start_x + 2*dx, start_y + 2*dy) for dx, dy in DIRECTIONS])

    while walls_list:
        wx, wy, nx, ny = walls_list.pop(random.randint(0, len(walls_list) - 1))
        if 0 < nx < MAZE_WIDTH and 0 < ny < MAZE_HEIGHT:
            if maze[ny][nx] == 'X':
                maze[wy][wx] = '.'
                maze[ny][nx] = '.'
                for dx, dy in DIRECTIONS:
                    walls_list.append((nx + dx, ny + dy, nx + 2*dx, ny + 2*dy))

    # Додаємо додаткові з'єднання для створення циклів
    for _ in range(level * 10):
        x = random.randrange(1, MAZE_WIDTH - 1, 2)
        y = random.randrange(1, MAZE_HEIGHT - 1, 2)
        if maze[y][x] == '.':
            direction = random.choice(DIRECTIONS)
            nx, ny = x + direction[0], y + direction[1]
            if 0 < nx < MAZE_WIDTH - 1 and 0 < ny < MAZE_HEIGHT - 1:
                if maze[ny][nx] == 'X':
                    maze[ny][nx] = '.'

    # Гарантуємо, що стартові позиції вільні
    positions = [
        (MAZE_WIDTH // 2, MAZE_HEIGHT // 2),  # Pac-Man
        (1, 1),  # Ghost 1
        (MAZE_WIDTH - 2, 1),  # Ghost 2
        (1, MAZE_HEIGHT - 2),  # Ghost 3
        (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)  # Ghost 4
    ]
    for x, y in positions:
        maze[y][x] = '.'

    return maze

# Перевірка зв'язності між стартовими позиціями
def is_maze_valid(maze, positions):
    for pos in positions[1:]:
        path = bfs(maze, positions[0], pos)
        if not path:
            return False
    return True

# Функція для створення рівня
def create_level(maze):
    walls = pygame.sprite.Group()
    global dots
    dots = pygame.sprite.Group()
    global power_pellets
    power_pellets = pygame.sprite.Group()
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            pos = (x * TILE_SIZE, y * TILE_SIZE)
            if tile == 'X':
                wall = Wall(pos)
                walls.add(wall)
                all_sprites.add(wall)
            elif tile == '.':
                dot = Dot(pos)
                dots.add(dot)
                all_sprites.add(dot)
    # Додаємо Power Pellets
    pellet_positions = [(1, 1), (MAZE_WIDTH - 2, 1), (1, MAZE_HEIGHT - 2), (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)]
    for x, y in pellet_positions:
        pellet = PowerPellet((x * TILE_SIZE, y * TILE_SIZE))
        power_pellets.add(pellet)
        all_sprites.add(pellet)
    return walls, dots, power_pellets

# Ініціалізація спрайтів та груп
all_sprites = pygame.sprite.Group()
ghosts = pygame.sprite.Group()

# Створення Pac-Man та привидів
pacman = PacManAI((screen_width // 2, screen_height // 2))
ghost1 = Ghost((TILE_SIZE + TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2), RED, 'dfs', role='chaser')
ghost2 = Ghost((screen_width - TILE_SIZE - TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2), PINK, 'bfs', role='ambusher')
ghost3 = Ghost((TILE_SIZE + TILE_SIZE // 2, screen_height - TILE_SIZE - TILE_SIZE // 2), CYAN, 'astar', role='patroller')
ghost4 = Ghost((screen_width - TILE_SIZE - TILE_SIZE // 2, screen_height - TILE_SIZE - TILE_SIZE // 2), ORANGE, 'heuristic', role='random')

ghosts.add(ghost1, ghost2, ghost3, ghost4)
all_sprites.add(pacman)
all_sprites.add(ghosts)

current_level = 1
max_levels = 5  # Кількість рівнів з наростанням складності

# Основний цикл гри
running = True
while running:
    # Генерація лабіринту з перевіркою зв'язності
    valid_maze = False
    while not valid_maze:
        maze = generate_maze(current_level)
        positions = [
            (MAZE_WIDTH // 2, MAZE_HEIGHT // 2),  # Pac-Man
            (1, 1),  # Ghost 1
            (MAZE_WIDTH - 2, 1),  # Ghost 2
            (1, MAZE_HEIGHT - 2),  # Ghost 3
            (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)  # Ghost 4
        ]
        valid_maze = is_maze_valid(maze, positions)
    walls, dots, power_pellets = create_level(maze)
    pacman.rect.center = (screen_width // 2, screen_height // 2)
    pacman.direction = (0, 0)
    pacman.powered_up = False
    pacman.powerup_timer = 0
    pacman.image = pacman.original_image.copy()
    pygame.draw.circle(pacman.image, YELLOW, (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
    pacman.maze = maze
    pacman.path = []
    for ghost in ghosts:
        # Встановлюємо початкові позиції привидів
        if ghost.algorithm == 'dfs':
            ghost.rect.center = (TILE_SIZE + TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2)
        elif ghost.algorithm == 'bfs':
            ghost.rect.center = (screen_width - TILE_SIZE - TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2)
        elif ghost.algorithm == 'astar':
            ghost.rect.center = (TILE_SIZE + TILE_SIZE // 2, screen_height - TILE_SIZE - TILE_SIZE // 2)
        elif ghost.algorithm == 'heuristic':
            ghost.rect.center = (screen_width - TILE_SIZE - TILE_SIZE // 2, screen_height - TILE_SIZE - TILE_SIZE // 2)
        ghost.initial_position = ghost.rect.center
        ghost.maze = maze
        ghost.path = []
        ghost.path_timer = 0
        ghost.needs_new_path = True
        ghost.frightened_mode = False
        ghost.speed = 1
        ghost.image = ghost.base_image.copy()
        ghost.target = None
        ghost.state = 'chase'

    level_running = True
    while level_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                level_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    # Перехід на наступний рівень при натисканні 'N'
                    current_level += 1
                    if current_level <= max_levels:
                        print("Перехід на рівень", current_level)
                    else:
                        print("Ви досягли максимального рівня!")
                        current_level = 1  # Починаємо знову
                    level_running = False

        pacman.update(walls)

        for ghost in ghosts:
            ghost.update(walls)

        game_timer.update()

        # Перевірка зіткнення Pac-Man з точками
        eaten_dots = pygame.sprite.spritecollide(pacman, dots, True)
        pacman.score += len(eaten_dots) * 10  # Кожна точка дає 10 балів
        if eaten_dots:
            pacman.notify_maze_changed()

        # Перевірка зіткнення з Power Pellets
        eaten_pellets = pygame.sprite.spritecollide(pacman, power_pellets, True)
        if eaten_pellets:
            pacman.powered_up = True
            pacman.powerup_timer = 600  # 10 секунд при 60 FPS
            pacman.image.fill((0, 0, 0, 0))
            pygame.draw.circle(pacman.image, (255, 215, 0), (TILE_SIZE//2, TILE_SIZE//2), TILE_SIZE//2)
            for ghost in ghosts:
                if ghost.state != 'respawn':
                    ghost.enter_frightened_mode()

        # Перевірка на зіткнення з привидами
        collided_ghost = pygame.sprite.spritecollideany(pacman, ghosts)
        if collided_ghost and collided_ghost.state != 'respawn':
            if pacman.powered_up and collided_ghost.state == 'frightened':
                # Pac-Man їсть привида
                collided_ghost.start_respawn()
                pacman.score += 200
            elif collided_ghost.state != 'frightened':
                # Pac-Man втрачає життя
                pacman.lives -= 1
                if pacman.lives <= 0:
                    print("Гру закінчено!")
                    running = False
                    level_running = False
                else:
                    # Перезапуск позиції Pac-Man
                    pacman.rect.center = (screen_width // 2, screen_height // 2)
                    pacman.direction = (0, 0)
                    pacman.path = []
                    # Перезапуск привидів
                    for ghost in ghosts:
                        ghost.rect.center = ghost.initial_position
                        ghost.path = []
                        ghost.path_timer = 0
                        ghost.needs_new_path = True
                        ghost.state = 'chase'
                        ghost.image = ghost.base_image.copy()
                        ghost.speed = 1

        # Перевірка на завершення рівня
        if len(dots) == 0:
            current_level += 1
            if current_level <= max_levels:
                print("Рівень пройдено!")
                level_running = False
            else:
                print("Ви виграли гру!")
                running = False
                level_running = False

        # Малювання
        screen.fill(BLACK)
        all_sprites.draw(screen)

        # Відображення рахунку та життів
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Рахунок: {pacman.score}", True, WHITE)
        lives_text = font.render(f"Життя: {pacman.lives}", True, WHITE)
        level_text = font.render(f"Рівень: {current_level}", True, WHITE)
        info_text = font.render("Натисніть 'N' для переходу на наступний рівень", True, WHITE)
        screen.blit(score_text, (5, 5))
        screen.blit(lives_text, (5, 30))
        screen.blit(level_text, (5, 55))
        screen.blit(info_text, (5, screen_height - 30))

        pygame.display.flip()
        clock.tick(60)

    # Очищення спрайтів для наступного рівня
    walls.empty()
    dots.empty()
    power_pellets.empty()
    all_sprites.empty()
    all_sprites.add(pacman)
    all_sprites.add(ghosts)

pygame.quit()
sys.exit()
