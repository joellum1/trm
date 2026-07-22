from pathlib import Path
import csv

from solve import solve_maze

WIDTH = 30
HEIGHT = 30


def load_maze(filename):

    with open(filename) as f:

        rows = [line.rstrip("\n") for line in f]

    rows = [r.ljust(WIDTH, "#")[:WIDTH] for r in rows]

    while len(rows) < HEIGHT:
        rows.append("#" * WIDTH)

    return rows


maze = load_maze("src/trm/data/custom/maze.txt")

solution, path = solve_maze(maze)

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

print("Dataset written.")
print(f"Optimal path length = {len(path)-1}")