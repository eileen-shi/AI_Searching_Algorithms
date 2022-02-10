# README

## Project Background
During a reconnaissance mission gone wrong, R2-D2 was attacked by Stormtroopers, leaving his executive control unit disconnected from his motor control unit.
Luckily, R2-D2’s motor control unit can still access his 9G-capable network card.
He just needs you to SSH into his motor control unit and guide him to the
rendezvous with C-3PO and Luke, but time is of the essence, so you must use
A* search to get him there as fast as possible. He just needs you to program and
run the A* search algorithm and integrate motor controls via his motor control
unit API.

## Main Concepts and Implementations
In this project, we compared differences between “uninformed” search
algorithms like BFS and DFS, and “informed” search algorithms like A* and Traveling Sales Person (TSP). The main difference between these algorithms are as following:

### **Breadth First Search**
A typical uninformed search algorithm, in which the root node is expanded first, then all the successors of the root node are expanded next, then their successors, and so on. BFS is complete even on infinite state spaces, always finds a solution with a minimal number of actions thus it is cost-optimal when all actions have the same cost. The issue is its exponential time and space complexity. 

### **Depth First Search**
Another uninformed search algorithm that always expands the deepest node in the frontier first. DFS is not cost-optimal and returns the first solution it finds even if it is not the cheapest. For finite state spaces, DFS works as a tree-like search and is efficient and complete. But in cyclic state spaces it can get stuck in an infinite loop. Since DFS doesn't maintain a reached table, its frontier is small and thus implement a linear space complexity.

### **A star Search**
The most common informed search algorithm, which is also a best-first search that consider both path cost from the initial state to node n and the estimated cost of the shortest path from node n to a goal state. A* search is complete and cost-optimal if the heauristic is admissible. 

### **Traveling Sales Person (TSP)**
TSP algorithm solves a search problem with multiple goals. The TSP method should return the shortest path which visits all the goal nodes.