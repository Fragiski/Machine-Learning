import java.util.*;

/*
* Deadlock detection 
* Used for boxes that are placed in a non-goal corner
*/

public class Deadlock {

    public static boolean isDeadlocked(Point p, Board b, Set<Point> boxes) {

        if (b.goals[p.r][p.c]) return false; // If the box is on a goal, not a deadlock

        return cornerDeadlock(p,b); // Returns whether the box is on a corner
    }


    //  Checks if the box is on a corner
    private static boolean cornerDeadlock(Point p, Board b) {

        boolean up = isWall(b, p.r - 1, p.c);
        boolean down = isWall(b, p.r + 1, p.c);
        boolean left = isWall(b, p.r, p.c -1);
        boolean right = isWall(b, p.r, p.c + 1);

        return (up && left) || (up && right) ||
                (down && left) || (down && right);

    }

    // Checks if the cell is a wall OR if it is out of bounds
    private static boolean isWall(Board b, int r, int c) {
        if (!b.inBounds(r,c)) return true;
        return b.walls[r][c];
    }
    
}
