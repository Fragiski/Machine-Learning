import java.util.*;

/*
* BFS used to compute the shortest walking distance of the player from one cell to another
*
*/
public class BFS {

    // Computes shortest distance from "start" to "target"
    public static int distance(Board b, Set<Point> boxes, Point start, Point target) {


        Queue<Point> q = new ArrayDeque<>(); // queue for BFS frontier
        Map<Point, Integer> dist = new HashMap<>(); // Stores each visited cell with its distance

        // Initializes BFS
        q.add(start);
        dist.put(start, 0);

        // Movements directions
        int[] dr = {-1,1,0,0};
        int[] dc = {0,0,-1,1};

        // BFS traversal
        while (!q.isEmpty()) {
            Point p = q.poll();
            int d = dist.get(p);

            if (p.equals(target))
                return d; // when target is reached

            // Explores neighbours
            for (int i = 0; i < 4; i++) {
                Point n = p.move(dr[i], dc[i]); // next cell

                if (!b.inBounds(n.r,n.c)) continue; // out of bounds
                if (b.walls[n.r][n.c]) continue;
                if (boxes.contains(n)) continue;  // box 
                if (dist.containsKey(n)) continue; // already visited

                dist.put(n, d + 1); // Marks visited
                q.add(n); // pushes to queue
            }
        }
        return -1; // if the player cannot reach the target 
    }
}
