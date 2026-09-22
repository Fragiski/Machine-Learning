import java.util.*;

/* 
 * Represents a Sokoban board 
 * Stores:
 * - the wall layout
 * - the goal positions
 * - the dimensions (rows,cols)
*/


public class Board {

    public final int rows, cols;
    public final boolean[][] walls;  // true if cell contains a wall
    public final boolean[][] goals;  // true if cell contains a goal


    // Constructs the board by parsing the input map lines
    public Board (List<String> lines) {

        rows = lines.size();
        cols = lines.stream().mapToInt(String::length).max().orElse(0);

        walls = new boolean [rows][cols];
        goals = new boolean [rows][cols];

        for (int r = 0; r < rows; r++) {

            String row = lines.get(r);
            for (int c = 0; c < row.length(); c++) {
                char ch = row.charAt(c);

                if (ch == '#') walls[r][c] = true;
                if (ch == '.' || ch == '*' || ch == '+') goals[r][c] = true;
                // (Player and boxes are handled in Main)
            }
        }

    }

    // Returns true if the given coordinates are inside the board boundaries
    public boolean inBounds(int r, int c) {
        return r >= 0 && r < rows && c >= 0 && c < cols; 
    }

    // Returns true if the cell is free for the player to walk in
    public boolean isFree(int r, int c, Set<Point> boxes) {
        if (!inBounds(r,c)) return false;
        return !walls[r][c] && !boxes.contains(new Point(r,c));
    }
    
}
