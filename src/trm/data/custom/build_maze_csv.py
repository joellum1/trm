from pathlib import Path
import csv

from solve import solve_maze, DEFAULT_REWARD_BONUS, DEFAULT_PENALTY_COST

WIDTH = 30
HEIGHT = 30

# Which maze to build, and how strongly reward/penalty tiles pull the route.
MAZE_FILE = "src/trm/data/custom/maze_objectives.txt"
REWARD_BONUS = DEFAULT_REWARD_BONUS
PENALTY_COST = DEFAULT_PENALTY_COST


def load_maze(filename):

    with open(filename) as f:

        rows = [line.rstrip("\n") for line in f]

    rows = [r.ljust(WIDTH, "#")[:WIDTH] for r in rows]

    while len(rows) < HEIGHT:
        rows.append("#" * WIDTH)

    return rows


maze = load_maze(MAZE_FILE)

solution, path, cost = solve_maze(
    maze,
    reward_bonus=REWARD_BONUS,
    penalty_cost=PENALTY_COST,
)

question = "\n".join(maze)
answer = "\n".join(solution)

rating = "custom"

Path("datasets").mkdir(exist_ok=True)

with open("data/custom.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "id",
        "q",
        "a",
        "r"
    ])

    writer.writerow([
        0,
        question,
        answer,
        rating
    ])

n_checkpoints = sum(row.count("C") for row in maze)
n_reward_total = sum(row.count("R") for row in maze)
n_penalty_total = sum(row.count("N") for row in maze)
n_reward_crossed = sum(row.count("r") for row in solution)
n_penalty_crossed = sum(row.count("n") for row in solution)

print("Dataset written.")
print(f"Path length (steps)     = {len(path) - 1}")
print(f"Total path cost         = {cost:.2f}")
print(f"Checkpoints visited     = {n_checkpoints}/{n_checkpoints} (mandatory)")
print(f"Reward tiles crossed    = {n_reward_crossed}/{n_reward_total}")
print(f"Penalty tiles crossed   = {n_penalty_crossed}/{n_penalty_total}")
