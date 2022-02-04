import pygame
import math
from queue import PriorityQueue
import random
from pygame import mouse

# display
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))  # create window
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


# square box for each node
class Node:
    
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.total_rows = total_rows
        self.x = row * width  # keep track of coordinates of box
        self.y = col * width
        self.color = WHITE
        self.neighbors = []

    def get_pos(self):
        return self.row, self.col
    
    def is_closed(self):
        return self.color == RED
    
    def is_open(self):
        return self.color == GREEN
    
    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE
    
    def make_closed(self):
        self.color = RED
    
    def make_open(self):
        self.color = GREEN
    
    def make_barrier(self):
        self.color = BLACK
    
    def make_start(self):
        self.color = ORANGE
    
    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < (self.total_rows - 1) and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < (self.total_rows - 1) and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    # compare function
    def __lt__(self, other):
        # comparing self with other, other spot 
        return False


# heuristics function, manhattan distance (quickest L distance)
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# heuristics function, euclidean distance
# def h(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def reconstruct_path(came_from, curr_node, draw):
    while curr_node in came_from:
        curr_node = came_from[curr_node]
        curr_node.make_path()
        draw()


def astar_algo(draw, grid, start, end):
    cnt = 0
    open_set = PriorityQueue()
    open_set.put((0, cnt, start))  # f_score = 0
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        curr_node = open_set.get()[2]
        open_set_hash.remove(curr_node)

        if curr_node == end:
            reconstruct_path(came_from, curr_node, draw)
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in curr_node.neighbors:
            temp_g_score = g_score[curr_node] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = curr_node
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

                if neighbor not in open_set_hash:
                    cnt += 1
                    open_set.put((f_score[neighbor], cnt, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if curr_node != start:
            curr_node.make_closed()
    
    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid


def make_random_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            rand_val = random.randint(1, 10)
            node = Node(i, j, gap, rows)
            if rand_val > 4:
                grid[i].append(node)
            else:
                node.make_barrier()
                grid[i].append(node)

    return grid


def make_set_grid(rows, width):
    grid = []
    gap = width // rows
    barrier_locs = [0]*int(rows**2*0.6) + [1]*int(rows**2*0.4)
    random.Random(10).shuffle(barrier_locs)
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            if barrier_locs[rows*i + j] == 0:
                grid[i].append(node)
            else:
                node.make_barrier()
                grid[i].append(node)
    
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    # draw nodes
    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(mouse_pos, rows, width):
    gap = width // rows
    y, x = mouse_pos
    row = y // gap
    col = x // gap  
    return row, col


def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)
    random_grid = make_random_grid(ROWS, width)
    set_grid = make_set_grid(ROWS, width)
    grid = set_grid

    start = None
    end = None

    run = True

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # mouse was pressed on left click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                curr_node = grid[row][col]
                if not start and curr_node != end:
                    start = curr_node
                    start.make_start()
                
                elif not end and curr_node != start:
                    end = curr_node
                    end.make_end()
                
                elif curr_node != start and curr_node != end:
                    curr_node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # mouse was pressed on right click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                curr_node = grid[row][col]
                curr_node.reset()
                if curr_node == start:
                    start = None
                
                elif curr_node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    astar_algo(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
    
    pygame.quit()

main(WIN, WIDTH)
