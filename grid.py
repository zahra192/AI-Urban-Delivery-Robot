import random

GRID_SIZE = 15
CELL_SIZE = 42
PADDING   = 10

ROAD     = 0
BUILDING = 1
TRAFFIC  = 2
DELIVERY = 3
BASE     = 4
PATH     = 5
EXPLORED = 6
ROBOT    = 7

COLORS = {
    ROAD:     "#1e2130",
    BUILDING: "#3a3f55",
    TRAFFIC:  "#5c3a1e",
    DELIVERY: "#0d7377",
    BASE:     "#6c3483",
    PATH:     "#f0a500",
    EXPLORED: "#1a4f6e",
    ROBOT:    "#e74c3c",
}

LABEL_COLORS = {
    BUILDING: "#7f8c8d",
    TRAFFIC:  "#e67e22",
    DELIVERY: "#1abc9c",
    BASE:     "#9b59b6",
}

class Grid:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cells      = [[ROAD]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.costs      = [[0]*GRID_SIZE    for _ in range(GRID_SIZE)]
        self.base       = None
        self.deliveries = []
        self._generate()

    def _generate(self):
        random.seed(random.randint(0, 99999))

        self.base = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        self.cells[self.base[0]][self.base[1]] = BASE

        n_buildings = int(GRID_SIZE * GRID_SIZE * 0.20)
        placed = 0
        while placed < n_buildings:
            r = random.randint(0, GRID_SIZE-1)
            c = random.randint(0, GRID_SIZE-1)
            if self.cells[r][c] == ROAD and (r, c) != self.base:
                self.cells[r][c] = BUILDING
                placed += 1

        n_traffic = int(GRID_SIZE * GRID_SIZE * 0.12)
        placed = 0
        while placed < n_traffic:
            r = random.randint(0, GRID_SIZE-1)
            c = random.randint(0, GRID_SIZE-1)
            if self.cells[r][c] == ROAD:
                self.cells[r][c] = TRAFFIC
                placed += 1

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                t = self.cells[r][c]
                if t in (ROAD, BASE):
                    self.costs[r][c] = random.randint(1, 5)
                elif t == TRAFFIC:
                    self.costs[r][c] = random.randint(10, 20)
                else:
                    self.costs[r][c] = 0

        self.deliveries = []
        while len(self.deliveries) < 5:
            r = random.randint(0, GRID_SIZE-1)
            c = random.randint(0, GRID_SIZE-1)
            if self.cells[r][c] in (ROAD, TRAFFIC) and (r, c) not in self.deliveries:
                self.deliveries.append((r, c))
                self.cells[r][c] = DELIVERY

    def neighbors(self, r, c):
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if self.cells[nr][nc] != BUILDING:
                    result.append((nr, nc, self.costs[nr][nc]))
        return result

    def is_traversable(self, r, c):
        return self.cells[r][c] != BUILDING