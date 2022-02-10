from typing import List, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
from queue import PriorityQueue
from itertools import permutations
import cProfile
import r2d2_hw2_gui as gui

# from spherov2 import scanner
# from spherov2.sphero_edu import SpheroEduAPI

Vertex = Tuple[int, int]
Edge = Tuple[Vertex, Vertex]

# Compare Different Searching Algorithms
class Graph:
    """A directed Graph representation"""

    def __init__(self, vertices: Set[Vertex], edges: Set[Edge]):
        self.vertices = vertices
        self.edges = edges
        self.dict_edges = defaultdict(list)
        for start, end in self.edges:
            self.dict_edges[start].append(end)

    def neighbors(self, u: Vertex) -> Set[Vertex]:
        """Return the neighbors of the given vertex u as a set"""
        return self.dict_edges[u]

    def bfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use BFS algorithm to find the path from start to goal in the given graph.
        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""
        ...
        node_visited = set()
        node_visited.add(start)
        trace = {}
        shortest_path = [goal]
        if start == goal:
            return shortest_path, node_visited
        frontier = [start]
        while frontier:
            node = frontier.pop(0)
            for vertex in self.neighbors(node):
                if vertex in node_visited:
                    continue
                else:
                    trace[tuple(vertex)] = node
                if vertex == goal:
                    while vertex != start:
                        shortest_path = [trace[tuple(vertex)]] + shortest_path
                        vertex = trace[tuple(vertex)]
                    node_visited.add(goal)
                    return shortest_path, node_visited

                else:
                    node_visited.add(vertex)
                    frontier.append(vertex)

    def dfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use DFS algorithm to find the path from start to goal in the given graph.
        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""
        ...
        node_visited = set()
        path = []
        visited = {}
        for vertex in self.vertices:
            visited[vertex] = False

        self.dfs_helper(start, goal, path, node_visited, visited)
        # if len(node_visited) and len(path):
        #     path.append(goal)
        return path, node_visited

    def dfs_helper(self, start, goal, path, node_visited, visited):
        visited[start] = True
        node_visited.add(start)
        path.append(start)
        if start == goal:
            return

        else:
            for node in self.neighbors(start):
                if not visited[node]:
                    self.dfs_helper(node, goal, path, node_visited, visited)
                    if goal in node_visited:
                        break
        if goal not in node_visited:
            path.pop()
            visited[start] = False

    def a_star(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use A* algorithm to find the path from start to goal in the given graph.
        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""

        if start == goal: return None
        start_r, start_c, goal_r, goal_c = start[0], start[1], goal[0], goal[1]

        frontier = PriorityQueue()
        frontier.put((heuristic(start_r, start_c, goal_r, goal_c), ([tuple(start)], 0)))

        came_from = {tuple(start): None}
        cost_so_far = {tuple(start): 0}

        while not frontier.empty():
            loc, cost = frontier.get()[1]
            current = loc[-1]

            if current == goal: break

            for new_loc in self.neighbors(current):
                new_cost = cost_so_far[current] + heuristic(current[0], current[1], new_loc[0], new_loc[1])
                if new_loc == tuple(goal):
                    return loc + [new_loc], list(came_from.keys())

                if new_loc not in cost_so_far or new_cost < cost_so_far[new_loc]:
                    cost_so_far[new_loc] = new_cost
                    priority = new_cost + heuristic(new_loc[0], new_loc[1], goal_r, goal_c)
                    frontier.put((priority, (loc + [new_loc], new_cost)))
                    came_from[new_loc] = current

    def tsp(self, start: Vertex, goals: Set[Vertex]) -> Tuple[Optional[List[Vertex]], Optional[List[Vertex]]]:
        """
        Traveling Sales Person (TSP) [20 points] Try to apply Traveling
        Sales Person (TSP) algorithm to solve a search problem with multiple
        goals. The tsp method should return the shortest path which visits all the
        goal nodes (note that your path should begin with the start node). You
        could use itertools.permutations to generate all the combinations of
        two target nodes and use A* to calculate the cost of each combination,
        then find the optimal order that has the shortest total cost.
        Use A* algorithm to find the path that begins at start and passes through all the goals in the given graph,
        in an order such that the path is the shortest.
        :return: a tuple (optimal_order, shortest_path),
                 where shortest_path is a list of vertices that represents the path from start that goes through all the
                 goals such that the path is the shortest; optimal_order is an ordering of goals that you visited in
                 order that results in the above shortest_path. Return (None, None) if no such path exists."""

        perm = permutations(goals)
        len_goals = len(goals)
        cost = np.inf

        all_comb = []
        for p in perm:
            all_comb.append([start]+list(p))

        for route in all_comb:
            cur_path = []
            path_cost = 0
            for j in range(len_goals):
                front, end = self.a_star(route[j], route[j+1])

                if not front:
                    break
                else:
                    for k in range(j+1):
                        if route[k] in front:
                            break

                    cur_cost = len(route)
                    path_cost += cur_cost
                    cur_path += route

            if path_cost < cost:
                cost, shortest_p, optimal, shortest = path_cost, route, route[1:], cur_path

        if cur_path:
            return optimal, shortest

        return None, None

def heuristic(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2) * 2


# Test
if __name__ == '__main__':
    V, E = gui.generate_random(12, 12)
    G = Graph(V, E)
    cProfile.run("print(G.tsp((6, 5), {(4, 1), (3, 3), (5, 9),(6,7),(9,10)}))")
