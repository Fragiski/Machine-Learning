# Sokoban Solver with A*

An automated solver for the Sokoban puzzle game written in Java. It uses the A* search algorithm combined with Breadth-First Search (BFS) to handle player movement and path validation.

## Overview

Sokoban is a transport puzzle where the player has to push boxes onto designated target squares. Boxes cannot be pulled, and moving a box into the wrong place can easily lead to a deadlock. This project searches for solutions by exploring valid box transitions while checking player reachability with BFS.

## How it Works

The core solver uses the A* algorithm with f = g + h evaluation. 

To keep the search space manageable, state hashing is based solely on the positions of the boxes rather than the player position. If two configurations share the same box placements, they are treated as the same state.

Before executing a push, a BFS search checks if the player can actually walk from their current position to the cell directly behind the box. If reachable, the actual path length is added to the transition cost.

The program includes two heuristics:
- h1: Calculates the sum of Manhattan distances from each box to the nearest target.
- h2: A greedy matching heuristic that matches each box to an available goal, providing a more balanced estimate.

Basic corner deadlock detection is also implemented to prune states where a box gets stuck against walls.

## File Structure

- Point.java: 2D coordinate class with custom equals and hashCode implementations.
- Board.java: Holds the static map, walls, and goal positions.
- State.java: Represents box locations, player position, and accumulated cost.
- BFS.java: Computes player reachability and step costs.
- Astar.java: Priority queue implementation of the A* search.
- Heuristics.java: Contains the h1 and h2 heuristic implementations.
- Deadlock.java: Corner deadlock checks.
- Main.java: Command line interface that reads maps and runs the solver.

## Board Legend

Level text files use the following format:

\# : Wall  

. : Target goal

@ : Player

$ : Box

\* : Box placed on target
  
" ": Empty floor

#### Requirements and Compilation

Requires JDK 17 or later. 

No external libraries are needed.

Compile all source files from the project folder

Run the program: java Main

When prompted, enter the file name of the level you want to test (for example level1.txt). The program will print the initial state, the solved state, along with push count, total moves, and execution time in milliseconds.

Academic Context
Developed as a group project, coursework for the "Artificial Intelligence" course, Academic Year 2025-26.
