World = [
    [0,0,0],
    [0,0,0]
]

directions = {
    "north": [0, 1],
    "east": [1, 0],
    "south": [0, -1],
    "west": [-1, 0],
    "nothing": [0,0]
}

direction_prob = {
    "north": ("north", "east", "west"),
    "east": ("east", "north", "south"),
    "south": ("south", "east", "west"),
    "west": ("west", "north", "south"),
    "nothing": ("nothing", "nothing", "nothing")
}

rewards = [
    [-0.1, -0.1, -0.05],
    [-0.1, -0.1, 1]
]


gamma = 0.999

epsilon = 0.01


delta = 22 #something bigger

delta_thresh = epsilon * (1-gamma) / gamma


def bellmenEquation(row, col):
    return 

iteration = 0

while(delta > delta_thresh):

    newWorld = [[-10,-10,-10],[-10,-10,-10]]
    iteration += 1

    for row in range(2):
        for col in range(3):

            if(row == 1 and col == 2):
                newWorld[row][col] = 1
                continue

            for direction in direction_prob.keys():
                dir_9, dir_05, dir_05b = direction_prob[direction]


                next_row_9 = row + directions[dir_9][0]
                next_col_9 = col + directions[dir_9][1]
                next_row_05 = row + directions[dir_05][0]
                next_col_05 = col + directions[dir_05][1]
                next_row_05b = row + directions[dir_05b][0]
                next_col_05b = col + directions[dir_05b][1]

                if next_row_9 < 0 or next_row_9 >= 2 or next_col_9 < 0 or next_col_9 >= 3:
                    next_row_9 = row
                    next_col_9 = col
                if next_row_05 < 0 or next_row_05 >= 2 or next_col_05 < 0 or next_col_05 >= 3:
                    next_row_05 = row
                    next_col_05 = col
                if next_row_05b < 0 or next_row_05b >= 2 or next_col_05b < 0 or next_col_05b >= 3:
                    next_row_05b = row
                    next_col_05b = col

                value_9 = rewards[next_row_9][next_col_9]
                if(next_row_9 != 1 or next_col_9 != 2):
                    value_9 += gamma * World[next_row_9][next_col_9]

                value_05 = rewards[next_row_05][next_col_05]
                if(next_row_05 != 1 or next_col_05 != 2):
                    value_05 += gamma * World[next_row_05][next_col_05]

                value_05b = rewards[next_row_05b][next_col_05b]
                if(next_row_05b != 1 or next_col_05b != 2):
                    value_05b += gamma * World[next_row_05b][next_col_05b]

                value = value_9 * 0.9 + value_05 * 0.05 + value_05b * 0.05
                newWorld[row][col] = max(newWorld[row][col], value)

    delta = max(abs(newWorld[row][col] - World[row][col]) for row in range(2) for col in range(3))
    World = newWorld

    for row in World:
        print([round(elem, 3) for elem in row])

    print("Iterations:", iteration, " Delta:", round(delta, 6), "\n")