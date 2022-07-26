import time
import random

import networkx as nx
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import pandas as pd

'''G = nx.grid_2d_graph(5, 5)

plt.figure(figsize=(10, 10))
pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(G, pos=pos,
        node_color='black',
        with_labels=False,
        node_size=1)
plt.show()'''
#settings-ish
width = 20
height = 20
width_offset = 0.5 #pls dont change
height_offset = 0.5 # pls dont change

# main maze points
maze_points = []  # (x,y) points
G = nx.Graph()
maze_edges = []  # last parameter is weight
pos = {}
fig, ax = plt.subplots()
labels = nx.get_edge_attributes(G, '')

# networkx pathfinding points -- contains all possible positions the AI can be in
ai_maze_points = []  # (x,y) points
ai_maze_edges = []  # last parameter is weight
ai_pos = {}
start_choice = None
start_edge = None
end_choice = None
end_edge = None
ai_start_choice = None
ai_start_edge = None
ai_end_choice = None
ai_end_edge = None


def add_edge_to_graph(graph, e1, e2, c, w):
    graph.add_edge(e1, e2, color=c, weight=w)


def replace_item_in_list(array, item_to_replace, item):
    i = array.index(item_to_replace)
    array = array[:i] + [item] + array[i + 1:]


def carve_maze(grid, size):
    output_grid = np.empty([size * 3, size * 3], dtype=str)
    output_grid[:] = '1'
    i = 0
    j = 0
    while i < size:
        w = i * 3 + 1
        while j < size:
            k = j * 3 + 1
            toss = grid[i, j]
            output_grid[w, k] = '0'
            if toss == 0 and k + 2 < size * 3:
                output_grid[w, k + 1] = '0'
                output_grid[w, k + 2] = '0'
            if toss == 1 and w - 2 >= 0:
                output_grid[w - 1, k] = '0'
                output_grid[w - 2, k] = '0'
            j = j + 1
        i = i + 1
        j = 0
    return output_grid


def create_maze_grid(points_arr, edge_arr, width, height, width_offset, height_offset):
    width_range = range(width + 1)
    height_range = range(height + 1)

    for h in height_range:
        for w in width_range:
            points_arr.append((w + width_offset, h + height_offset))

    for p in points_arr:  # Horizontal Lines
        p_new = points_arr.index(p)

        if p_new == 0:
            edge_arr.append((p_new, p_new + 1, 1))
        elif (p_new + 1) % (width + 1) != 0:
            edge_arr.append((p_new, p_new + 1, 1))

    for j in range(height + 1):
        saved_points = []

        for p in points_arr:
            if p[0] == j + height_offset:
                saved_points.append(p)

        saved_points_range = range(len(saved_points))

        for k in saved_points_range:
            if (saved_points[k])[1] < height:
                edge_arr.append((points_arr.index(saved_points[k]), points_arr.index(saved_points[k + 1]), 1))

        saved_points = []


def weighted_random(choices, weights):
    return random.choices(choices, cum_weights=weights, k=1)


def find_commonality_between_lists(list1, list2):
    for i in list1:
        if i in list2:
            return True


def generate_maze(graph, points_arr, edges_arr, ai_points_arr, ai_edges_arr, height, width, width_offset,
                  height_offset):
    # create random start and end
    global start_choice
    global start_edge
    global end_choice
    global end_edge

    global ai_start_choice
    global ai_start_edge
    global ai_end_choice
    global ai_end_edge

    bottom_width = points_arr[0:width]
    top_width = points_arr[-(width + 1):-1]

    start_choice = random.choice(bottom_width)
    start_edge = (start_choice, points_arr[points_arr.index(start_choice) + 1])
    end_choice = random.choice(top_width)
    end_edge = (end_choice, points_arr[points_arr.index(end_choice) + 1])

    ai_start_choice = (start_choice[0] + width_offset, start_choice[1] + height_offset)
    ai_end_choice = (end_choice[0] + width_offset, end_choice[1] + height_offset)

    # creates start and end walls
    '''

    # create paths
    path = []'''

    # my own maze generation system -- still needs lots of work
    '''start_node_path = []
    end_node_path = []

    current_start_node = ai_start_choice
    current_end_node = ai_end_choice
    start_choices = [(0, 1), (-1, 0), (1, 0)]
    end_choices = [(0, -1), (-1, 0), (1, 0)]

    weights = [1, random.randint(2, 10), random.randint(2, 10)]

    start_node_path.append(current_start_node)
    end_node_path.append(current_end_node)

    for i in range(random.randint(5, 14)):
        def find_node(direction, current_node):
            node_index = current_node

            match direction[0]:
                case (0, 1):
                    if (current_node[0], current_node[1] + 1) in ai_points_arr:
                        node_index = (current_node[0], current_node[1] + 1)

                case (0, -1):
                    if (current_node[0], current_node[1] - 1) in ai_points_arr:
                        node_index = (current_node[0], current_node[1] - 1)

                case (-1, 0):
                    if (current_node[0] - 1, current_node[1]) in ai_points_arr:
                        node_index = (current_node[0] - 1, current_node[1])

                case (1, 0):
                    if (current_node[0] + 1, current_node[1]) in ai_points_arr:
                        node_index = (current_node[0] + 1, current_node[1])

            return node_index
        current_start_node = find_node(weighted_random(start_choices, weights), current_start_node)
        current_end_node = find_node(weighted_random(end_choices, weights), current_end_node)
        start_node_path.append(current_start_node)
        end_node_path.append(current_end_node)'''

    # create maze using only numpy binary tree method -- still needs testing.
    '''n = 1
    p = 0.4
    size = 7
    grid = np.random.binomial(n, p, size=(size, size))
    first_row = grid[0]
    first_row[first_row == 1] = 0
    grid[0] = first_row

    for i in range(1, size):
        grid[i, size - 1] = 1

    path = carve_maze(grid, size)
    array_remove_points = []
    array_remove_edges = []
    #print(path)

    current_iteration = 0
    for i in path:
        i_reverse = i[::-1]
        adder = 0
        for n in i_reverse:
            if n == "0":
                array_remove_points.append(ai_points_arr[-(current_iteration + 1)])
            current_iteration = current_iteration + 1


    print(array_remove_points)

    return [i for i in ai_points_arr if i not in array_remove_points]'''

    # This algorithm is a randomized version of Prim's algorithm.
    #+
    # Start with a grid full of walls.
    # Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
        # While there are walls in the list:
            # Pick a random wall from the list. If only one of the cells that the wall divides is visited, then:
                # Make the wall a passage and mark the unvisited cell as part of the maze.
                # Add the neighboring walls of the cell to the wall list.
            # Remove the wall from the list.
    '''Note that simply running classical Prim's on a graph with random edge weights would create mazes stylistically 
    identical to Kruskal's, because they are both minimal spanning tree algorithms. 
    Instead, this algorithm introduces stylistic variation because the edges closer to the starting 
    point have a lower effective weight.'''
    visited_cells = []  # cell list -- i used another grid inside of the main maze grid to create the possible positions that the program can choose inside of the maze
    unvisited_cells = []
    maze_walls = []
    delete_maze_walls = []

    def update_unvisited_cells():
        nonlocal unvisited_cells
        unvisited_cells = [i for i in ai_points_arr if i not in visited_cells]

    def check_cells_that_wall_divides(edge, edges_list, edges_points_list, cell_points_list):

        reference_point = ()
        cell_point1 = ()
        cell_point2 = ()
        nonlocal edge_direction

        if edges_points_list[edge[0]][0] == edges_points_list[edge[1]][0]:

            edge_direction = "vertical"
            reference_point = edges_points_list[edge[0]] if edges_points_list[edge[0]][1] < edges_points_list[edge[1]][1] else edges_points_list[edge[1]]  # top-most point
            cell_point1 = (reference_point[0] + width_offset,
                           reference_point[1] + height_offset)  # right side point of vertical dividing edge
            cell_point2 = (reference_point[0] - width_offset,
                           reference_point[1] + height_offset)  # left side point of vertical dividing edge
        elif edges_points_list[edge[0]][1] == edges_points_list[edge[1]][1]:

            edge_direction = "horizontal"
            reference_point = edges_points_list[edge[0]] if edges_points_list[edge[0]][0] < edges_points_list[edge[1]][0] else edges_points_list[edge[1]]  # left-most point
            cell_point1 = (reference_point[0] + width_offset,
                           reference_point[1] + height_offset)  # top point of horizontal dividing edge
            cell_point2 = (reference_point[0] + width_offset,
                           reference_point[1] - height_offset)  # bottom point of horizontal dividing edge
        else:
            raise ValueError("Edges are probably invalid and something bad happened")
        return cell_point1, cell_point2, edge_direction

    def find_walls_from_cell(cell, excluded_walls):  # returns a list of edges surrounding a cell -- includes exclusion of a list of walls
        walls = []

        top_right_corner = ()
        top_left_corner = ()
        bottom_right_corner = ()
        bottom_left_corner = ()

        top_wall = ()
        bottom_wall = ()
        left_wall = ()
        right_wall = ()

        wall_storage = []

        if (cell[0] + width_offset, cell[1] + height_offset) in points_arr:  # check if top right corner of cell exists
            top_right_corner = (int(cell[0] + width_offset), int(cell[1] + height_offset))
            if (cell[0] - width_offset, cell[1] + height_offset) in points_arr:  # check if top left corner of cell exists
                top_left_corner = (int(cell[0] - width_offset), int(cell[1] + height_offset))
                if (cell[0] + width_offset, cell[1] - height_offset) in points_arr:  # check if bottom right corner of cell exists
                    bottom_right_corner = (int(cell[0] + width_offset), int(cell[1] - height_offset))
                    if (cell[0] - width_offset, cell[1] - height_offset) in points_arr:  # check if bottom left corner of cell exists
                        bottom_left_corner = (int(cell[0] - width_offset), int(cell[1] - height_offset))
                        if (points_arr.index(top_right_corner), points_arr.index(top_left_corner), 1) in edges_arr:
                            top_wall = (points_arr.index(top_right_corner), points_arr.index(top_left_corner), 1)
                            if not (top_left_corner[1] == 0 and top_right_corner[1] == 0):
                                if not (top_left_corner[1] == height and top_right_corner[1] == height):
                                    wall_storage.append(top_wall)
                                else:
                                    top_wall = ()
                        elif (points_arr.index(top_left_corner), points_arr.index(top_right_corner), 1) in edges_arr:
                            top_wall = (points_arr.index(top_left_corner), points_arr.index(top_right_corner), 1)
                            if not (top_left_corner[1] == 0 and top_right_corner[1] == 0):
                                if not (top_left_corner[1] == height and top_right_corner[1] == height):
                                    wall_storage.append(top_wall)
                                else:
                                    top_wall = ()

                        if (points_arr.index(top_right_corner), points_arr.index(bottom_right_corner), 1) in edges_arr:
                            right_wall = (points_arr.index(top_right_corner), points_arr.index(bottom_right_corner), 1)
                            if not (bottom_right_corner[0] == 0 and top_right_corner[0] == 0):
                                if not (bottom_right_corner[0] == width and top_right_corner[0] == width):
                                    wall_storage.append(right_wall)
                                else:
                                    right_wall = ()
                        elif (points_arr.index(bottom_right_corner), points_arr.index(top_right_corner), 1) in edges_arr:
                            right_wall = (points_arr.index(bottom_right_corner), points_arr.index(top_right_corner), 1)
                            if not (bottom_right_corner[0] == 0 and top_right_corner[0] == 0):
                                if not (bottom_right_corner[0] == width and top_right_corner[0] == width):
                                    wall_storage.append(right_wall)
                                else:
                                    right_wall = ()

                        if (points_arr.index(bottom_left_corner), points_arr.index(top_left_corner), 1) in edges_arr:
                            left_wall = (points_arr.index(bottom_left_corner), points_arr.index(top_left_corner), 1)
                            if not (bottom_left_corner[0] == 0 and top_left_corner[0] == 0):
                                if not (bottom_left_corner[0] == width and top_left_corner[0] == width):
                                    wall_storage.append(left_wall)
                                else:
                                    left_wall = ()
                        elif (points_arr.index(top_left_corner), points_arr.index(bottom_left_corner), 1) in edges_arr:
                            left_wall = (points_arr.index(top_left_corner), points_arr.index(bottom_left_corner), 1)
                            if not (bottom_left_corner[0] == 0 and top_left_corner[0] == 0):
                                if not (bottom_left_corner[0] == width and top_left_corner[0] == width):
                                    wall_storage.append(left_wall)
                                else:
                                    left_wall = ()

                        if (points_arr.index(bottom_left_corner), points_arr.index(bottom_right_corner), 1) in edges_arr:
                            bottom_wall = (points_arr.index(bottom_left_corner), points_arr.index(bottom_right_corner), 1)
                            if not (bottom_left_corner[1] == 0 and bottom_right_corner[1] == 0):
                                if not (bottom_left_corner[1] == height and bottom_right_corner[1] == height):
                                    wall_storage.append(bottom_wall)
                                else:
                                    bottom_wall = ()
                        elif (points_arr.index(bottom_right_corner), points_arr.index(bottom_left_corner), 1) in edges_arr:
                            bottom_wall = (points_arr.index(bottom_right_corner), points_arr.index(bottom_left_corner), 1)
                            if not (bottom_left_corner[1] == 0 and bottom_right_corner[1] == 0):
                                if not (bottom_left_corner[1] == height and bottom_right_corner[1] == height):
                                    wall_storage.append(bottom_wall)
                                else:
                                    bottom_wall = ()

        walls = [i for i in wall_storage if i not in excluded_walls]
        '''print(top_right_corner)
        print(top_left_corner)
        print(bottom_right_corner)
        print(bottom_left_corner)
        print(top_wall)
        print(bottom_wall)
        print(left_wall)
        print(right_wall)
        print(walls)
        print(excluded_walls)'''
        return walls
    ram_walls = find_walls_from_cell(ai_start_choice, [])
    print("starting walls = " + str(ram_walls))
    for i in ram_walls:
        maze_walls.append(i)
    visited_cells.append(ai_start_choice)
    update_unvisited_cells()
    while maze_walls:
        #wall = weighted_random(maze_walls, np.ones(len(maze_walls)))
        wall = random.choice(maze_walls)
        #print(wall)
        cell1, cell2, edge_direction = check_cells_that_wall_divides(wall, edges_arr, points_arr, ai_points_arr)
        current_unvisited_cell = ()
        if not (cell1 in visited_cells and cell2 in visited_cells):
            if cell1 not in visited_cells:
                current_unvisited_cell = cell1
            elif cell2 not in visited_cells:
                current_unvisited_cell = cell2

        if current_unvisited_cell != ():  # make wall a passage
            delete_maze_walls.append(wall)
            maze_walls_temp = find_walls_from_cell(current_unvisited_cell, wall)
            maze_wall_temp_points = []
            for i in maze_walls_temp:
                maze_wall_temp_points.append((points_arr[i[0]], points_arr[i[1]]))
            #print("cell1 = " + str(cell1), "cell2 = " + str(cell2), "current cell = " + str(current_unvisited_cell), "wall = " + str(wall), "edge direction = " + str(edge_direction), "point1 of wall = " + str(points_arr[wall[0]]), "point2 of wall = " + str(points_arr[wall[1]]), "walls around cell = " + str(maze_wall_temp_points))

            for i in maze_walls_temp:
                if i not in maze_walls:
                    maze_walls.append(i)  # add cell's edges to wall list
            visited_cells.append(current_unvisited_cell)  # cell is now visited
            update_unvisited_cells()

        maze_walls.remove(wall)


    print("visited_cells = " + str(visited_cells))
    print("unvisited_cells = " + str(unvisited_cells))
    print("delete_maze_walls = " + str(delete_maze_walls))
    edges_arr.remove((points_arr.index(end_choice), points_arr.index(end_choice) + 1, 1))
    edges_arr.remove((points_arr.index(start_choice), points_arr.index(start_choice) + 1, 1))
    return delete_maze_walls


def remove_blocked_ai_edges(maze_edges, maze_points, ai_edges, ai_points, width_offset, height_offset):
    remove_ai_edges = []

    for ai_point in ai_points:
        if (ai_point[0] + 1, ai_point[1]) in ai_points: # check if right cell exists
            if (ai_point[0] + width_offset, ai_point[1] + height_offset) in maze_points and (ai_point[0] + width_offset, ai_point[1] - height_offset) in maze_points: # check if maze points exist
                if (maze_points.index((ai_point[0] + width_offset, ai_point[1] + height_offset)), maze_points.index((ai_point[0] + width_offset, ai_point[1] - height_offset)), 1) in maze_edges or (maze_points.index((ai_point[0] + width_offset, ai_point[1] - height_offset)), maze_points.index((ai_point[0] + width_offset, ai_point[1] + height_offset)), 1) in maze_edges: # check if vertical maze edge is in the way
                    # check if ai edge exists and append accordingly
                    if (ai_points.index(ai_point), ai_points.index((ai_point[0] + 1, ai_point[1])), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index(ai_point), ai_points.index((ai_point[0] + 1, ai_point[1])), 1))
                    elif (ai_points.index((ai_point[0] + 1, ai_point[1])), ai_points.index(ai_point), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index((ai_point[0] + 1, ai_point[1])), ai_points.index(ai_point), 1))

        if (ai_point[0] - 1, ai_point[1]) in ai_points:  # check if left cell exists
            if (ai_point[0] - width_offset, ai_point[1] + height_offset) in maze_points and (ai_point[0] - width_offset, ai_point[1] - height_offset) in maze_points: # check if maze points exist
                if (maze_points.index((ai_point[0] - width_offset, ai_point[1] + height_offset)), maze_points.index((ai_point[0] - width_offset, ai_point[1] - height_offset)), 1) in maze_edges or (maze_points.index((ai_point[0] - width_offset, ai_point[1] - height_offset)), maze_points.index((ai_point[0] - width_offset, ai_point[1] + height_offset)), 1) in maze_edges: # check if vertical maze edge is in the way
                    # check if ai edge exists and append accordingly
                    if (ai_points.index((ai_point), ai_points.index((ai_point[0] - 1, ai_point[1]))), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index(ai_point), ai_points.index((ai_point[0] - 1, ai_point[1])), 1))
                    elif (ai_points.index((ai_point[0] - 1, ai_point[1])), ai_points.index(ai_point), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index((ai_point[0] - 1, ai_point[1])), ai_points.index(ai_point), 1))


        if (ai_point[0], ai_point[1] + 1) in ai_points:  # check if top cell exists
            if (ai_point[0] + width_offset, ai_point[1] + height_offset) in maze_points and (ai_point[0] - width_offset, ai_point[1] + height_offset) in maze_points: # check if maze points exist
                if (maze_points.index((ai_point[0] + width_offset, ai_point[1] + height_offset)), maze_points.index((ai_point[0] - width_offset, ai_point[1] + height_offset)), 1) in maze_edges or (maze_points.index((ai_point[0] - width_offset, ai_point[1] + height_offset)), maze_points.index((ai_point[0] + width_offset, ai_point[1] + height_offset)), 1) in maze_edges: # check if vertical maze edge is in the way
                    # check if ai edge exists and append accordingly
                    if (ai_points.index(ai_point), ai_points.index((ai_point[0], ai_point[1] + 1)), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index(ai_point), ai_points.index((ai_point[0], ai_point[1] + 1)), 1))
                    elif (ai_points.index((ai_point[0], ai_point[1] + 1)), ai_points.index(ai_point), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index((ai_point[0], ai_point[1] + 1)), ai_points.index(ai_point), 1))


        if (ai_point[0], ai_point[1] - 1) in ai_points:  # check if bottom cell exists
            if (ai_point[0] + width_offset, ai_point[1] - height_offset) in maze_points and (ai_point[0] - width_offset, ai_point[1] - height_offset) in maze_points: # check if maze points exist
                if (maze_points.index((ai_point[0] - width_offset, ai_point[1] - height_offset)), maze_points.index((ai_point[0] + width_offset, ai_point[1] - height_offset)), 1) in maze_edges or (maze_points.index((ai_point[0] + width_offset, ai_point[1] - height_offset)), maze_points.index((ai_point[0] - width_offset, ai_point[1] - height_offset)), 1) in maze_edges: # check if vertical maze edge is in the way
                    # check if ai edge exists and append accordingly
                    if (ai_points.index(ai_point), ai_points.index((ai_point[0], ai_point[1] - 1)), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index(ai_point), ai_points.index((ai_point[0], ai_point[1] - 1)), 1))
                    elif (ai_points.index((ai_point[0], ai_point[1] - 1)), ai_points.index(ai_point), 1) in ai_edges:
                        remove_ai_edges.append((ai_points.index((ai_point[0], ai_point[1] - 1)), ai_points.index(ai_point), 1))

    return remove_ai_edges


def show_edges(graph, edges, points, ai_points, path):
    new_path = []

    for i in range(len(path) - 1):
        if (ai_points.index(path[i]), ai_points.index(path[i + 1]), 1) in edges:
            new_path.append((ai_points.index(path[i]), ai_points.index(path[i + 1]), 1))
        elif (ai_points.index(path[i + 1]), ai_points.index(path[i]), 1) in edges:
            new_path.append((ai_points.index(path[i + 1]), ai_points.index(path[i]), 1))
    print("path = " + str(new_path))

    for i in range(len(edges)):
        if 0 <= edges[i][0] < len(points) and 0 <= edges[i][1] < len(points):
            if edges[i] in new_path:
                add_edge_to_graph(graph, points[edges[i][0]], points[edges[i][1]], 'blue', edges[i][2])
            elif isinstance(points[edges[i][0]][0], float):
                add_edge_to_graph(graph, points[edges[i][0]], points[edges[i][1]], 'lightblue', edges[i][2])
            else:
                add_edge_to_graph(graph, points[edges[i][0]], points[edges[i][1]], 'black', edges[i][2])



def state_position(points, graph, axis):
    color_list = ['red', 'blue', 'green', 'yellow', 'purple']
    positions = {point: point for point in points}
    edges = G.edges()
    nodes = G.nodes()

    edge_colors = [G[u][v]['color'] for u, v in edges]
    nx.draw(graph, pos=positions, node_size=0, edge_color=edge_colors, node_color='black', ax=axis)


create_maze_grid(maze_points, maze_edges, width, height, 0, 0)
create_maze_grid(ai_maze_points, ai_maze_edges, width - 1, height - 1, height_offset, width_offset)
deleted_edges = generate_maze(G, maze_points, maze_edges, ai_maze_points, ai_maze_edges, width, height, height_offset, width_offset)
temp_maze_edges = maze_edges
maze_edges = [i for i in temp_maze_edges if i not in deleted_edges]
ai_deleted_edges = remove_blocked_ai_edges(maze_edges, maze_points, ai_maze_edges, ai_maze_points, 0.5, 0.5)
temp_ai_maze_edges = ai_maze_edges
ai_maze_edges = [i for i in temp_ai_maze_edges if i not in ai_deleted_edges]
show_edges(G, ai_maze_edges, ai_maze_points, ai_maze_points, [])
shortest_path = nx.shortest_path(G, ai_start_choice, (ai_end_choice[0], ai_end_choice[1] - 1))
print(shortest_path)
G.remove_edges_from(ai_maze_edges)
G.remove_nodes_from(ai_maze_points)
print(G.nodes, G.edges)
show_edges(G, ai_maze_edges, ai_maze_points, ai_maze_points, shortest_path)
show_edges(G, maze_edges, maze_points, ai_maze_points, [])
state_position(maze_points + ai_maze_points, G, ax)
print("maze points" + str(maze_points))
print("ai maze points" + str(ai_maze_points))
print("maze edges" + str(maze_edges))
print("ai maze edges" + str(ai_maze_edges))


# state_position(ai_maze_points, G, ax)

'''nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)'''  # coord and weight labels

plt.axis("on")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()

