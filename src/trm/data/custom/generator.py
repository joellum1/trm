import random
import csv
from pathlib import Path

from solve import solve_maze, DEFAULT_REWARD_BONUS, DEFAULT_PENALTY_COST


WIDTH = 30
HEIGHT = 30


TRAIN_SAMPLES = 500
TEST_SAMPLES = 100


OUTPUT_DIR = Path("data/maze-custom")


CHECKPOINTS = (1, 4)
REWARDS = (1, 5)
PENALTIES = (1, 5)


# -------------------------------------------------
# Maze generation
# -------------------------------------------------

def generate_empty_grid():
    return [
        ["#" for _ in range(WIDTH)]
        for _ in range(HEIGHT)
    ]


def neighbours(x, y):

    dirs = [
        (2,0),
        (-2,0),
        (0,2),
        (0,-2)
    ]

    result = []

    for dx, dy in dirs:

        nx = x + dx
        ny = y + dy

        if (
            0 <= nx < WIDTH and
            0 <= ny < HEIGHT
        ):
            result.append((nx,ny))

    random.shuffle(result)

    return result



def carve_maze(grid, x, y):

    grid[y][x] = " "

    for nx, ny in neighbours(x,y):

        if grid[ny][nx] == "#":

            # carve connecting wall
            grid[
                y + (ny-y)//2
            ][
                x + (nx-x)//2
            ] = " "

            carve_maze(
                grid,
                nx,
                ny
            )



def generate_maze():

    grid = generate_empty_grid()


    # must be odd coordinates
    start_x = 1
    start_y = 1

    carve_maze(
        grid,
        start_x,
        start_y
    )


    return grid



# -------------------------------------------------
# Add objectives
# -------------------------------------------------

def random_empty_cell(grid):

    cells = []

    for y in range(HEIGHT):
        for x in range(WIDTH):

            if grid[y][x] == " ":
                cells.append((x,y))

    return random.choice(cells)



def place_objectives(grid):

    sx, sy = random_empty_cell(grid)
    gx, gy = random_empty_cell(grid)


    while (gx,gy)==(sx,sy):
        gx,gy=random_empty_cell(grid)


    grid[sy][sx]="S"
    grid[gy][gx]="G"



    # checkpoints

    for _ in range(
        random.randint(*CHECKPOINTS)
    ):

        x,y=random_empty_cell(grid)
        grid[y][x]="C"



    # rewards

    for _ in range(
        random.randint(*REWARDS)
    ):

        x,y=random_empty_cell(grid)

        if grid[y][x]==" ":
            grid[y][x]="R"



    # penalties

    for _ in range(
        random.randint(*PENALTIES)
    ):

        x,y=random_empty_cell(grid)

        if grid[y][x]==" ":
            grid[y][x]="N"



    return grid



# -------------------------------------------------
# Dataset generation
# -------------------------------------------------

def generate_dataset(filename, count):


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        filename,
        "w",
        newline=""
    ) as f:


        writer = csv.writer(f)


        writer.writerow([
            "id",
            "q",
            "a",
            "r"
        ])



        for i in range(count):


            while True:

                maze = generate_maze()

                maze = place_objectives(
                    maze
                )


                try:

                    solution, path, cost = solve_maze(
                        maze,
                        reward_bonus=DEFAULT_REWARD_BONUS,
                        penalty_cost=DEFAULT_PENALTY_COST,
                    )

                    break

                except Exception:

                    # regenerate invalid mazes
                    continue



            question = "\n".join(
                "".join(row)
                for row in maze
            )


            answer = "\n".join(
                "".join(row)
                for row in solution
            )


            writer.writerow([
                i,
                question,
                answer,
                "custom"
            ])



            if i % 100 == 0:
                print(
                    f"{filename}: {i}/{count}"
                )



# -------------------------------------------------

if __name__ == "__main__":


    generate_dataset(
        OUTPUT_DIR/"train.csv",
        TRAIN_SAMPLES
    )


    generate_dataset(
        OUTPUT_DIR/"test.csv",
        TEST_SAMPLES
    )


    print("Finished")