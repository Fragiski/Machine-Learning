import java.util.*;
/*
* A* solver for Sokoban using f = g + h
* g = walking + pushes (walking via BFS)
* h = heuristic distance to goals
* States store player, boxes, and g
* Deadlocks (e.g., box in non-goal corner) are pruned
*/
public class Astar {

    private static class Node implements Comparable<Node> {
        State s;
        int f;

        Node(State s, int f) { 
            this.s = s; 
            this.f = f;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(f, o.f);
        }
    }

    public static List<State> solve(Board b, State start) {
        
        //open: priority queue ordered by f = g + h
        PriorityQueue<Node> open = new PriorityQueue<>(); 
        
        //gScore: the lowest g-cost found so far for this box configuration
        Map<State, Integer> gScore = new HashMap<>();

        //parent pointers: for reconstructing final solution path
        Map<State, State> parent = new HashMap<>();

        //Initial heuristic
        int h = Heuristics.h2(start, b);
        open.add(new Node(start, h));
        gScore.put(start, 0);

        //Pushing directions
        int[] dr = {-1,1,0,0};
        int[] dc = {0,0,-1,1};


        //Main A* Loop
        while (!open.isEmpty()) {

            Node n = open.poll();
            State cur =n.s;
            
            //If all boxes are on goals then solved
            if (cur.isGoal(b)) 
                return reconstruct(cur, parent);

            //Try to push each box in 4 possible directions
            for (Point box : cur.boxes) {
                for (int k = 0; k < 4; k++) {

                    //Player must be behind the box
                    Point behind = box.move(-dr[k], -dc[k]);
                    Point ahead = box.move( dr[k], dc[k]);

                    //Out of board limits
                    if (!b.inBounds(behind.r,behind.c) || 
                        !b.inBounds(ahead.r,ahead.c))
                        continue;

                    //Is direction ahead a wall
                    if (b.walls[ahead.r][ahead.c]) 
                        continue;
                    //Is direction ahead is a goal
                    if (cur.boxes.contains(ahead))
                         continue;
                    ////Is direction ahead is a goal
                    if (Deadlock.isDeadlocked(ahead, b, cur.boxes))
                         continue;


                    //Player must be able to walk and go behind cell
                    int walkDist = BFS.distance(b, cur.boxes, cur.player, behind);
                    if (walkDist == -1) 
                        continue;


                    //After each push, build a new state
                    Set<Point> newBoxes = new HashSet<>(cur.boxes);
                    newBoxes.remove(box);
                    newBoxes.add(ahead);

                    //Total cost = old g + walking steps + 1 push
                    int newG = cur.g + walkDist + 1;
 
                    //Player ends up at the previous box position
                    State next = new State(box, newBoxes, newG);

                    int tentativeG = newG;

                    if (!gScore.containsKey(next) || tentativeG < gScore.get(next)) {
                       gScore.put(next, tentativeG);
                       
                       //f = g + h
                       int F = tentativeG + Heuristics.h2(next, b);

                       parent.put(next, cur);
                       open.add(new Node(next, F));
                    }
                }
            }   
        }

        return null;
    }

    //Reconstructs the full solution path by following parent pointers
    private static List<State> reconstruct (State goal, Map<State, State> parent) {
        List<State> path = new ArrayList<>();
        State cur = goal;

        while (cur != null) {
            path.add(cur);
            cur = parent.get(cur);
        }

        Collections.reverse(path);
        return path;
    }
}