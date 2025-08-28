import heapq
import matplotlib.pyplot as plt
import time
import math

GRID_SIZE = 10

# Create grid and add obstacles
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
obstacles = [(0, 2), (1, 3), (2, 1), (2, 6), (3, 2), (3, 6), (4, 3), (4, 7),
             (5, 1), (5, 5), (6, 4), (6, 8), (7, 2), (7, 6), (8, 3), (8, 7)]

for obs in obstacles:
    grid[obs[0]][obs[1]] = 1

def is_valid_cell(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x][y] == 0

def get_neighbors(cell):
    x, y = cell
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    return [(x+dx, y+dy) for dx, dy in dirs if is_valid_cell(x+dx, y+dy)]

def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def reconstruct_path(parent, current):
    path = [current]
    while current in parent and parent[current] is not None:
        current = parent[current]
        path.append(current)
    return list(reversed(path))

def A_Star(start, goal):
    open_list = []
    heapq.heappush(open_list, (euclidean_distance(start, goal), start))

    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}
    parent = {start: None}
    closed_set = set()
    visited_nodes = 0

    while open_list:
        cost, current = heapq.heappop(open_list)

        if current in closed_set:
            continue

        visited_nodes += 1

        if current == goal:
            return reconstruct_path(parent, current), visited_nodes

        closed_set.add(current)

        for nb in get_neighbors(current):
            if nb in closed_set:
                continue

            temp_g = g_score[current] + 1
            if nb not in g_score or temp_g < g_score[nb]:
                g_score[nb] = temp_g
                parent[nb] = current
                f_score[nb] = temp_g + euclidean_distance(nb, goal)
                heapq.heappush(open_list, (f_score[nb], nb))

    return None, visited_nodes

# Run A*
start_node = (0,0)
goal_node = (GRID_SIZE-1, GRID_SIZE-1)

start_time = time.time()
path_A_Star, visited_A_Star = A_Star(start_node, goal_node)
end_time = time.time()
time_A_Star = end_time - start_time

print(f"A* Time: {time_A_Star:.6f} seconds, Nodes Visited: {visited_A_Star}")

# Plotting
fig, ax = plt.subplots()
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if grid[x][y] == 1:
            ax.scatter(x, y, color='black', s=100)

if path_A_Star:
    px, py = zip(*path_A_Star)
    ax.plot(px, py, color='lime', linewidth=3)

ax.scatter(start_node[0], start_node[1], color='green', s=100, label='Start')
ax.scatter(goal_node[0], goal_node[1], color='red', s=100, label='Goal')

ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.grid(True)
ax.set_xlim(-1, GRID_SIZE)
ax.set_ylim(-1, GRID_SIZE)
ax.legend()
plt.show()
