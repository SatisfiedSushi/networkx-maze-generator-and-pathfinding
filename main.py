import time
import random

import networkx as nx
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

#main maze points
maze_points = []  # (x,y) points
G = nx.Graph()
maze_edges = []  # last parameter is weight
pos = {}
fig, ax = plt.subplots()
labels = nx.get_edge_attributes(G, '')

#networkx pathfinding points -- contains all possible positions the AI can be in
ai_maze_points = []  # (x,y) points
ai_maze_edges = []  # last parameter is weight
ai_pos = {}
start_choice = None
start_edge = None
end_choice = None
end_edge = None

def add_edge_to_graph(graph, e1, e2, w):
    graph.add_edge(e1, e2, color='b', weight=w)


def replace_item_in_list(array, item_to_replace, item):
    i = array.index(item_to_replace)
    array = array[:i] + [item] + array[i + 1:]


def create_maze_grid(points_arr, edge_arr, width, height, width_offset, height_offset):
    width_range = range(width + 1)
    height_range = range(height + 1)

    for h in height_range:
        for w in width_range:
            points_arr.append((w + width_offset, h + height_offset))

    for p in points_arr: #Horizontal Lines
        p_new = points_arr.index(p)

        if p_new == 0:
            edge_arr.append((p_new, p_new + 1, 1))
        elif (p_new+1) % (width + 1) != 0:
            edge_arr.append((p_new, p_new + 1, 1))

    for j in range(height + 1):
        saved_points = []

        for p in points_arr:
            if p[0] == j + height_offset:
                saved_points.append(p)

        saved_points_range = range(len(saved_points))

        for k in saved_points_range:
            if (saved_points[k])[1] < height:
                edge_arr.append((points_arr.index(saved_points[k]), points_arr.index(saved_points[k + 1]), 0))

        saved_points = []


def generate_maze(points_arr, edges_arr, ai_points_arr, ai_edges_arr, height, width):
    #create random start and end
    global start_choice
    global start_edge

    global end_choice
    global end_edge

    bottom_width = points_arr[0:width]
    top_width = points_arr[-(width + 1):-1]


    start_choice = random.choice(bottom_width)
    start_edge = (start_choice, points_arr[points_arr.index(start_choice) + 1])

    end_choice = random.choice(top_width)
    end_edge = (end_choice, points_arr[points_arr.index(end_choice) + 1])

    edges_arr.remove((points_arr.index(end_choice), points_arr.index(end_choice) + 1, 1))
    edges_arr.remove((points_arr.index(start_choice), points_arr.index(start_choice) + 1, 1))

    #create paths



def show_edges(graph, edges, points):
    for i in range(len(edges)):
        add_edge_to_graph(graph, points[edges[i][0]], points[edges[i][1]], edges[i][2])


def state_position(points, graph, axis):
    positions = {point: point for point in points}
    nx.draw(graph, pos=positions, node_size=10, node_color='black', ax=axis)


create_maze_grid(maze_points, maze_edges, 10, 10, 0, 0)
generate_maze(maze_points, maze_edges, ai_maze_points, ai_maze_edges, 10, 10)
create_maze_grid(ai_maze_points, ai_maze_edges, 9, 9, 0.5, 0.5)
show_edges(G, ai_maze_edges, ai_maze_points)
show_edges(G, maze_edges, maze_points)
state_position(maze_points + ai_maze_points, G, ax)
print("maze points" + str(maze_points))
print("ai maze points" + str(ai_maze_points))
print("maze edges" + str(maze_edges))
print("ai maze edges" + str(ai_maze_edges))


#state_position(ai_maze_points, G, ax)

'''nx.draw_networkx_labels(G, pos=pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)'''  # coord and weight labels
plt.axis("on")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()
