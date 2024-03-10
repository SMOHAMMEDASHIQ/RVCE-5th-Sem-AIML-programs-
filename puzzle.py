def generate_nextstate(data, level):
    #  four directions {up, down, left, right}
    x, y = find(data, '_')
    val_list = [[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y]]
    children = []
    for i in val_list:
        child = shuffle(data, x, y, i[0], i[1])
        if child is not None:
            children.append((child, level + 1, 0))
    return children


def shuffle(puz, x1, y1, x2, y2):
    if 0 <= x2 < len(puz) and 0 <= y2 < len(puz):
        temp_puz = [row[:] for row in puz]
        temp = temp_puz[x2][y2]
        temp_puz[x2][y2] = temp_puz[x1][y1]
        temp_puz[x1][y1] = temp
        return temp_puz
    else:
        return None





def find(puz, x):
    
    for i in range(0, len(puz)):
        for j in range(0, len(puz)):
            if puz[i][j] == x:
                return i, j


def h(start, goal):
    
    temp = 0
    for i in range(0, len(start)):
        for j in range(0, len(start)):
            if start[i][j] != goal[i][j] and start[i][j] != '_':
                temp += 1
    return temp


def f(start, goal, level):
    # Heuristic function to calculate Heuristic value f(x) = h(x) + g(x)
    return h(start, goal) + level


def start_game(size):
    
    print("Enter the start state matrix \n")
    start = accept(size)
    print("Enter the goal state matrix \n")
    goal = accept(size)
    start = (start, 0, 0)
    start_fval = f(start[0], goal, start[2])
    
    open_list = [(start, start_fval)]
    closed_list = []
    print("\n\n")
    while True:
        cur = open_list[0][0]
        print("==================================================\n")
        for i in cur[0]:
            for j in i:
                print(j, end=" ")
            print("")
        
        if h(cur[0], goal) == 0:
            break
        for child_data, child_level, _ in generate_nextstate(cur[0], start[2]):
            child_fval = f(child_data, goal, child_level)
            open_list.append(((child_data, child_level, 0), child_fval))
        closed_list.append(open_list[0])
        del open_list[0]
        # sort the open list based on f value
        open_list.sort(key=lambda x: x[1], reverse=False)


def accept(size):
    
    puz = []
    for i in range(0, size):
        temp = input().split(" ")
        puz.append(temp)
    return puz



puzzle_size = 3
start_game(puzzle_size)
