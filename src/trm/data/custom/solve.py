from collections import deque
from copy import deepcopy

DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]


def find_tile(grid, tile):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == tile:
                return (r, c)
    raise ValueError(f"{tile} not found")


def solve_maze(grid):
    """
    grid is a list of strings.

    Returns:
        solved_grid (list[str])
        path (list[(r,c)])
    """

    grid = [list(r) for r in grid]

    start = find_tile(grid, "S")
    goal = find_tile(grid, "G")

    q = deque([start])

    parent = {}

    visited = {start}

    while q:

        r, c = q.popleft()

        if (r, c) == goal:
            break

        for dr, dc in DIRS:

            nr = r + dr
            nc = c + dc

            if not (0 <= nr < len(grid)):
                continue

            if not (0 <= nc < len(grid[0])):
                continue

            if grid[nr][nc] == "#":
                continue

            if (nr, nc) in visited:
                continue

            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)

            q.append((nr, nc))

    if goal not in parent:
        raise RuntimeError("No path found")

    path = []

    node = goal

    while node != start:
        path.append(node)
        node = parent[node]

    path.append(start)

    path.reverse()

    solved = deepcopy(grid)

    for r, c in path:

        if solved[r][c] == " ":
            solved[r][c] = "o"

    solved = ["".join(r) for r in solved]

    return solved, path