import collections
import time
import matplotlib.pyplot as plt

GRID_SIZE = 10

# Initialize grid and obstacles
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
obstacles = [(0, 2), (1, 3), (2, 1), (2, 6), (3, 2), (3, 6), (4, 3), (4, 7),
             (5, 1), (5, 5), (6, 4), (6, 8), (7, 2), (7, 6), (8, 3), (8, 7)]


for x, y in obstacles:
    grid[x][y] = 1

def is_valid_cell(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x][y] == 0

def get_neighbors(cell):
    x, y = cell
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    return [(x+dx, y+dy) for dx,dy in directions if is_valid_cell(x+dx, y+dy)]

def reconstruct_path(parent, current):
    path = [current]
    while current in parent and parent[current] is not None:
        current = parent[current]
        path.append(current)
    return path[::-1]

def BFS(start, goal):
    queue = collections.deque([start])
    parent = {start: None}
    visited = set([start])

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(parent, current), visited

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None, visited

# Start and goal positions
start_node, goal_node = (0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)

# Measure BFS runtime
t0 = time.perf_counter()
path_bfs, visited_bfs = BFS(start_node, goal_node)
t1 = time.perf_counter()

print(f"BFS Time: {t1 - t0:.6f} seconds, Nodes Visited: {len(visited_bfs)}")

# Plot BFS path
fig, ax = plt.subplots()
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if grid[x][y] == 1:
            ax.scatter(x, y, color='black', s=100)

if path_bfs:
    px, py = zip(*path_bfs)
    ax.plot(px, py, color='lime', linewidth=3, label='BFS Path')

ax.scatter(0, 0, color='green', s=100, label='Start (0,0)')
ax.scatter(GRID_SIZE - 1, GRID_SIZE - 1, color='red', s=100, label='Goal (9,9)')
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.grid(True)
ax.set_xlim(-1, GRID_SIZE)
ax.set_ylim(-1, GRID_SIZE)
ax.legend()
plt.show()
