import java.util.*;

/*
*   h1: For each box, use Manhattan distance to the nearest goal
*   h2: Greedy used to match boxes to goals
*/
public class Heuristics {
    
    //h1: sum of Manhattan distances from each box to the closest goal
    public static int h1(State s, Board b) {
        int H = 0;
        for (Point box : s.boxes) {
            int best = Integer.MAX_VALUE;
            for (int r = 0; r < b.rows; r++)
                for (int c = 0; c < b.cols; c++)
                    if (b.goals[r][c])
                        best = Math.min (best, Math.abs(box.r - r) + Math.abs(box.c - c));
            H += best;        
        }
        return H;
    }

    //h2: assign each box to the nearest unused goal
    public static int h2(State s, Board b) {
        List<Point> goalList = new ArrayList<>();
        for (int r = 0; r < b.rows; r++)
            for (int c = 0; c < b.cols; c++)
                if (b.goals[r][c])
                    goalList.add(new Point(r,c));

        boolean[] used = new boolean[goalList.size()];
        int H = 0;

        for (Point box : s.boxes) {
            int bestD = Integer.MAX_VALUE;
            int bestIdx = -1;

            for (int i = 0; i < goalList.size(); i++) {
                if (used[i]) continue;
                Point g = goalList.get(i);
                int dist = Math.abs(box.r - g.r) + Math.abs(box.c - g.c);
                if (dist < bestD) {
                    bestD = dist;
                    bestIdx = i;
                }
            }

            used[bestIdx] = true;
            H += bestD;
        }
        return H;               
    }
}
