import numpy as np

from trm.evaluation.maze_checker import (
    check, summarize, CHARSET, VALID_SHORTEST, VALID_LONGER, BROKEN,
)

# Two routes S -> G: along the top (4 steps) or round the bottom (8 steps).
LOOP = [
    "#######",
    "#S   G#",
    "# ### #",
    "#     #",
    "#######",
]
LOOP_SHORT = [
    "#######",
    "#SoooG#",
    "# ### #",
    "#     #",
    "#######",
]
LOOP_LONG = [
    "#######",
    "#S   G#",
    "#o###o#",
    "#ooooo#",
    "#######",
]

# Two equal-length routes: top-right or bottom-left.
SQUARE = [
    "#######",
    "#S    #",
    "# ### #",
    "#    G#",
    "#######",
]
SQUARE_TOP = [
    "#######",
    "#Soooo#",
    "# ###o#",
    "#    G#",
    "#######",
]
SQUARE_BOTTOM = [
    "#######",
    "#S    #",
    "#o### #",
    "#ooooG#",
    "#######",
]


def codes(v):
    return [c for c, _ in v.reasons]


def test_label_is_valid_shortest_and_exact():
    v = check(LOOP, LOOP_SHORT, LOOP_SHORT)
    assert v.category == VALID_SHORTEST and v.exact
    assert v.pred_steps == v.opt_steps == 4


def test_alternate_shortest_route_is_valid_shortest_not_exact():
    v = check(SQUARE, SQUARE_BOTTOM, SQUARE_TOP)
    assert v.category == VALID_SHORTEST
    assert v.exact is False


def test_longer_route():
    v = check(LOOP, LOOP_LONG)
    assert v.category == VALID_LONGER
    assert codes(v) == ["longer"]
    assert (v.pred_steps, v.opt_steps) == (8, 4)


def test_both_sides_of_loop_is_branch():
    both = ["#######", "#SoooG#", "#o###o#", "#ooooo#", "#######"]
    v = check(LOOP, both)
    assert v.category == BROKEN and codes(v) == ["branch"]


def test_gap_is_disconnected():
    gap = ["#######", "#So oG#", "# ### #", "#     #", "#######"]
    v = check(LOOP, gap)
    assert v.category == BROKEN and "disconnected" in codes(v)


def test_stray_fragment():
    stray = ["#######", "#SoooG#", "# ### #", "#  o  #", "#######"]
    v = check(LOOP, stray)
    assert v.category == BROKEN and codes(v) == ["stray_cells"]


def test_path_on_wall():
    maze = ["#####", "#S#G#", "#####"]
    v = check(maze, ["#####", "#SoG#", "#####"])
    assert v.category == BROKEN and "path_on_wall" in codes(v)


def test_dead_end_checkpoint_spur_is_valid():
    maze = ["#######", "#S   G#", "### ###", "###C###", "#######"]
    label = ["#######", "#SoooG#", "###o###", "###C###", "#######"]
    v = check(maze, label, label)
    assert v.category == VALID_SHORTEST
    assert v.pred_steps == 8   # 4 along + 2 down + 2 back up


def test_unreached_checkpoint():
    maze = ["#######", "#S   G#", "### ###", "###C###", "#######"]
    pred = ["#######", "#SoooG#", "### ###", "###C###", "#######"]
    v = check(maze, pred)
    assert v.category == BROKEN and "missed_checkpoint" in codes(v)


def test_dead_end_spur_without_checkpoint_is_branch():
    maze = ["#######", "#S   G#", "### ###", "#######"]
    pred = ["#######", "#SoooG#", "###o###", "#######"]
    v = check(maze, pred)
    assert v.category == BROKEN and codes(v) == ["branch"]


def test_wrong_marker_on_reward_tile():
    maze = ["#######", "#S R G#", "#######"]
    assert check(maze, ["#######", "#SoroG#", "#######"]).category == VALID_SHORTEST
    v = check(maze, ["#######", "#SoooG#", "#######"])
    assert v.category == BROKEN and codes(v) == ["wrong_marker"]


def test_erased_start():
    v = check(LOOP, ["#######", "# oooG#", "# ### #", "#     #", "#######"])
    assert v.category == BROKEN
    assert codes(v)[:2] == ["tile_altered", "start_missing"]


def test_penalty_detour_is_shortest_by_cost():
    # Straight through N: 2 steps, cost (1+3) + 1 = 5. Round the bottom: 4 steps, cost 4.
    maze = ["#####", "#SNG#", "#   #", "#####"]
    through = ["#####", "#SnG#", "#   #", "#####"]
    around = ["#####", "#SNG#", "#ooo#", "#####"]
    assert check(maze, around).category == VALID_SHORTEST
    v = check(maze, through)
    assert v.category == VALID_LONGER and v.pred_steps < v.opt_steps


def test_token_id_arrays_round_trip():
    ids = {ch: i + 1 for i, ch in enumerate(CHARSET)}
    enc = lambda rows: np.array([[ids[ch] for ch in row] for row in rows]).reshape(-1)
    # 5x7 is not square, so pad to 7x7 with walls
    pad = lambda rows: rows + ["#######"] * (7 - len(rows))
    v = check(enc(pad(SQUARE)), enc(pad(SQUARE_BOTTOM)), enc(pad(SQUARE_TOP)))
    assert v.category == VALID_SHORTEST and v.exact is False
    pred = enc(pad(SQUARE_BOTTOM))
    pred[0] = 0  # PAD
    assert "bad_token" in codes(check(enc(pad(SQUARE)), pred))


def test_summarize():
    vs = [check(LOOP, LOOP_SHORT, LOOP_SHORT), check(LOOP, LOOP_LONG, LOOP_SHORT),
          check(LOOP, ["#######", "#So oG#", "# ### #", "#     #", "#######"], LOOP_SHORT)]
    s = summarize(vs)
    assert s["valid_shortest"] == s["valid_longer"] == s["broken"] == 1 / 3
    assert s["broken_primary/disconnected"] == 1 / 3
    assert s["exact"] == 1 / 3
