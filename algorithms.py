import time
import heapq
import math
from collections import deque

def reconstruct(came_from, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

def bfs(grid, start, goal):
    queue     = deque([start])
    visited   = {start}
    came_from = {start: None}
    explored  = []
    t0 = time.perf_counter()
    while queue:
        r, c = queue.popleft()
        explored.append((r, c))
        if (r, c) == goal:
            path = reconstruct(came_from, start, goal)
            cost = sum(grid.costs[nr][nc] for nr, nc in path[1:])
            return path, cost, explored, time.perf_counter()-t0
        for nr, nc, _ in grid.neighbors(r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    return None, 0, explored, time.perf_counter()-t0

def dfs(grid, start, goal):
    stack     = [start]
    visited   = set()
    came_from = {start: None}
    explored  = []
    t0 = time.perf_counter()
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        explored.append((r, c))
        if (r, c) == goal:
            path = reconstruct(came_from, start, goal)
            cost = sum(grid.costs[nr][nc] for nr, nc in path[1:])
            return path, cost, explored, time.perf_counter()-t0
        for nr, nc, _ in grid.neighbors(r, c):
            if (nr, nc) not in visited:
                came_from[(nr, nc)] = (r, c)
                stack.append((nr, nc))
    return None, 0, explored, time.perf_counter()-t0

def ucs(grid, start, goal):
    heap      = [(0, start)]
    g_cost    = {start: 0}
    came_from = {start: None}
    explored  = []
    t0 = time.perf_counter()
    while heap:
        cost, (r, c) = heapq.heappop(heap)
        if cost > g_cost.get((r, c), float('inf')):
            continue
        explored.append((r, c))
        if (r, c) == goal:
            path = reconstruct(came_from, start, goal)
            return path, cost, explored, time.perf_counter()-t0
        for nr, nc, w in grid.neighbors(r, c):
            new_cost = cost + w
            if new_cost < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_cost
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, (nr, nc)))
    return None, 0, explored, time.perf_counter()-t0

def _euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def greedy(grid, start, goal):
    heap      = [(_euclidean(start, goal), start)]
    visited   = set()
    came_from = {start: None}
    explored  = []
    t0 = time.perf_counter()
    while heap:
        _, (r, c) = heapq.heappop(heap)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        explored.append((r, c))
        if (r, c) == goal:
            path = reconstruct(came_from, start, goal)
            cost = sum(grid.costs[nr][nc] for nr, nc in path[1:])
            return path, cost, explored, time.perf_counter()-t0
        for nr, nc, _ in grid.neighbors(r, c):
            if (nr, nc) not in visited:
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (_euclidean((nr,nc), goal), (nr, nc)))
    return None, 0, explored, time.perf_counter()-t0

def astar(grid, start, goal):
    h         = lambda n: _manhattan(n, goal)
    heap      = [(h(start), 0, start)]
    g_cost    = {start: 0}
    came_from = {start: None}
    explored  = []
    t0 = time.perf_counter()
    while heap:
        f, cost, (r, c) = heapq.heappop(heap)
        if cost > g_cost.get((r, c), float('inf')):
            continue
        explored.append((r, c))
        if (r, c) == goal:
            path = reconstruct(came_from, start, goal)
            return path, cost, explored, time.perf_counter()-t0
        for nr, nc, w in grid.neighbors(r, c):
            new_cost = cost + w
            if new_cost < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_cost
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost + h((nr,nc)), new_cost, (nr, nc)))
    return None, 0, explored, time.perf_counter()-t0

ALGORITHMS = {
    "BFS":    bfs,
    "DFS":    dfs,
    "UCS":    ucs,
    "Greedy": greedy,
    "A*":     astar,
}