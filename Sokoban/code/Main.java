import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {


    public static void main(String[] args) throws Exception {

        Scanner sc = new Scanner(System.in);
        String filename;


        // Level file input
        while (true) {

            System.out.print("Enter level filename (e.g. level.txt): ");
            filename = sc.nextLine().trim();


            File f = new File(filename);

            // File not found
            if (!f.exists() || !f.isFile()) {

                System.out.println("ERROR: File not found.Please try again.\n");
                continue;
            }

            break;


        }


        // Reads level file and creates board
        List<String> lines = Files.readAllLines(Paths.get(filename));
        Board b = new Board(lines);

        Point player = null;
        Set<Point> boxes = new HashSet<>();


        for (int r=0; r < lines.size(); r++) {

            String row = lines.get(r);
            for(int c = 0; c< row.length(); c++) {
                char ch = row.charAt(c);

                if (ch == '@' || ch == '+') player = new Point(r,c);
                if (ch == '$' || ch == '*') boxes.add(new Point(r,c));
            }
        }


        // Player not found
        if (player == null) {

            System.out.println("ERROR: Level contains no player '@'.");
            return;
        }

        State start = new State(player, boxes, 0);


        // Prints initial state
        System.out.println("\n=== INITIAL STATE (" + filename + ") ===\n");
        printState(b,start);
        System.out.println("-----------------------------------------\n");

        
        // Runs A* and keeps track of time
        long t0 = System.currentTimeMillis();
        List<State> sol = Astar.solve(b,start);
        long t1 = System.currentTimeMillis();

        // No solution found
        if (sol == null) {

            System.out.println("NO SOLUTION FOUND");
            return;
        }


        // Prints solution info
        System.out.println("\n=== SOLVED ===");
        System.out.println("\nLevel: " + filename);
        System.out.println("Time: " + (t1 - t0) + "ms");

        // Pushes
        int pushes = sol.size() -1;
        System.out.println("Pushes: " + pushes);

        // Moves
        int moves = sol.get(sol.size() - 1).g;
        System.out.println("Moves: " + moves);

        System.out.println("\n-------------------------------\n");


        // Prints final state
        System.out.println("\n=== FINAL STATE ===\n");
        printState(b,sol.get(sol.size() -1));
        System.out.println("-------------------------------\n");

    }


    // Printing-state function
    private static void printState(Board b, State s) {

        for(int r=0; r < b.rows; r++) {

            for(int c=0; c < b.cols; c++) {

                Point p = new Point(r,c);

                if (b.walls[r][c])         {System.out.print("#"); continue; }
                if (s.player.equals(p))    {System.out.print("@"); continue; }
                if (s.boxes.contains(p))   {

                    System.out.print(b.goals[r][c] ? "*" : "$");
                    continue;
                
                }
                if (b.goals[r][c])     {System.out.print("."); }
                else     {System.out.print(" "); }

            }

            System.out.println();
        }
        System.out.println();
        
    }


}
