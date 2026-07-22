"""
Maze solver with support for objective tiles:

    #   wall            impassable
    (space) empty       cost 1 to cross
    S   start
    G   goal
    C   checkpoint       MUST be crossed at some point on the route
    R   reward tile      optional, cheaper than a normal tile to cross
    N   penalty tile     optional, more expensive than a normal tile to cross
                         (still passable -- not a wall)

The solved grid marks the path taken. Empty cells on the path become 'o',
same as before. Reward/penalty tiles that were actually crossed become
lowercase ('r' / 'n') so the model can distinguish "this reward tile was
worth taking" from "this reward tile was in the maze but skipped".
Checkpoints don't need a separate visited marker: by construction every
'C' in a valid solution is always on the path.

This is no longer a plain BFS shortest path. With weighted tiles and a
set of mandatory checkpoints, the problem becomes: find the minimum-cost
route from S to G that visits every checkpoint, in whichever order is
cheapest. That's solved here as:

    1. Weighted Dijkstra from every key point (S, each checkpoint, G) to
       every other key point, respecting walls and tile costs.
    2. Held-Karp dynamic programming over subsets of checkpoints to find
       the cheapest visiting order (exact, not a nearest-neighbour
       heuristic).
    3. Stitch the resulting point-to-point segments into one path and
       overlay it onto the grid.
"""

import heapq
from copy import deepcopy

DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]

WALL = "#"
EMPTY = " "
START = "S"
GOAL = "G"
CHECKPOINT = "C"
REWARD = "R"
PENALTY = "N"

# Tuning knobs for how strongly reward/penalty tiles pull the route.
# Kept < 1.0 / > 0.0 so all edge costs stay positive (Dijkstra requires this).
DEFAULT_REWARD_BONUS = 0.5   # a reward tile costs (1 - bonus) to cross
DEFAULT_PENALTY_COST = 3.0   # a penalty tile costs (1 + this) to cross


def find_tile(grid, tile):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == tile:
                return (r, c)
    raise ValueError(f"{tile} not found")


def find_all(grid, tile):
    return [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == tile
    ]


def tile_cost(tile, reward_bonus, penalty_cost):
    if tile == REWARD:
        return max(1.0 - reward_bonus, 0.01)
    if tile == PENALTY:
        return 1.0 + penalty_cost
    return 1.0


def _dijkstra(grid, source, reward_bonus, penalty_cost):
    """Single-source weighted shortest paths avoiding walls.

    Returns (dist, parent) dicts keyed by (r, c).
    """

    h, w = len(grid), len(grid[0])

    dist = {source: 0.0}
    parent = {}
    visited = set()
    pq = [(0.0, source)]

    while pq:
        d, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        r, c = node

        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr][nc] == WALL:
                continue

            step_cost = tile_cost(grid[nr][nc], reward_bonus, penalty_cost)
            nd = d + step_cost
            nxt = (nr, nc)

            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                parent[nxt] = node
                heapq.heappush(pq, (nd, nxt))

    return dist, parent


def _reconstruct(parent, source, target):
    if target == source:
        return [source]
    if target not in parent:
        return None

    path = [target]
    node = target
    while node != source:
        node = parent[node]
        path.append(node)
    path.reverse()

    return path


def solve_maze(grid, reward_bonus=DEFAULT_REWARD_BONUS, penalty_cost=DEFAULT_PENALTY_COST):
    """
    grid: list[str]

    Finds the minimum-cost route from S to G that visits every mandatory
    checkpoint tile, taking reward/penalty tiles into account wherever the
    route happens to pass them.

    Returns:
        solved_grid (list[str])  -- input grid with the chosen route overlaid
        path (list[(r, c)])      -- full route, start to goal, in order
        cost (float)             -- total cost of the route
    """

    grid = [list(r) for r in grid]

    start = find_tile(grid, START)
    goal = find_tile(grid, GOAL)
    checkpoints = find_all(grid, CHECKPOINT)

    nodes = [start] + checkpoints + [goal]
    n = len(nodes)
    goal_idx = n - 1
    k = len(checkpoints)  # checkpoints occupy node indices 1..k

    # All-pairs shortest paths between key nodes only (S, checkpoints, G).
    dist_matrix = [[float("inf")] * n for _ in range(n)]
    path_cache = {}

    for i, src in enumerate(nodes):
        dist, parent = _dijkstra(grid, src, reward_bonus, penalty_cost)
        for j, dst in enumerate(nodes):
            if dst in dist:
                dist_matrix[i][j] = dist[dst]
                path_cache[(i, j)] = _reconstruct(parent, src, dst)

    if k == 0:
        best_cost = dist_matrix[0][goal_idx]
        if best_cost == float("inf"):
            raise RuntimeError("No path found")
        best_order = []
    else:
        # Held-Karp: dp[(mask, last)] = min cost to start at S, visit exactly
        # the checkpoints in `mask`, ending at checkpoint `last`.
        full_mask = (1 << k) - 1
        dp = {}
        dp_parent = {}

        for c_idx in range(1, k + 1):
            d = dist_matrix[0][c_idx]
            if d < float("inf"):
                dp[(1 << (c_idx - 1), c_idx)] = d

        for mask in range(1, full_mask + 1):
            for last in range(1, k + 1):
                if not (mask & (1 << (last - 1))):
                    continue
                cur = dp.get((mask, last))
                if cur is None:
                    continue
                for nxt in range(1, k + 1):
                    bit = 1 << (nxt - 1)
                    if mask & bit:
                        continue
                    d = dist_matrix[last][nxt]
                    if d == float("inf"):
                        continue
                    new_mask = mask | bit
                    new_cost = cur + d
                    if new_cost < dp.get((new_mask, nxt), float("inf")):
                        dp[(new_mask, nxt)] = new_cost
                        dp_parent[(new_mask, nxt)] = last

        best_cost = float("inf")
        best_last = None
        for last in range(1, k + 1):
            cur = dp.get((full_mask, last))
            d = dist_matrix[last][goal_idx]
            if cur is None or d == float("inf"):
                continue
            total = cur + d
            if total < best_cost:
                best_cost = total
                best_last = last

        if best_last is None:
            raise RuntimeError("No route visiting all checkpoints found")

        order = []
        mask, last = full_mask, best_last
        while last is not None:
            order.append(last)
            prev = dp_parent.get((mask, last))
            mask ^= (1 << (last - 1))
            last = prev
        order.reverse()
        best_order = order

    # Stitch the chosen segments (S -> checkpoints in best order -> G) into
    # a single path.
    route_nodes = [0] + best_order + [goal_idx]
    full_path = [nodes[0]]

    for a, b in zip(route_nodes[:-1], route_nodes[1:]):
        segment = path_cache.get((a, b))
        if segment is None:
            raise RuntimeError("No path found")
        full_path.extend(segment[1:])

    # Overlay the route onto the grid.
    solved = deepcopy(grid)

    for r, c in full_path:
        tile = solved[r][c]
        if tile == EMPTY:
            solved[r][c] = "o"
        elif tile == REWARD:
            solved[r][c] = "r"
        elif tile == PENALTY:
            solved[r][c] = "n"
        # WALL can't appear here; START / GOAL / CHECKPOINT stay as-is.

    solved = ["".join(r) for r in solved]

    return solved, full_path, best_cost
