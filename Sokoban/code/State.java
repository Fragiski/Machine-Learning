import java.util.*;

/**
 * Represents a Sokoban state 
 * Consists of:
 * - the player's current position
 * - the set of box positions
 * - the cumulative cost g (walking + pushes so far)
 */


public class State {


    public final Point player;
    public final Set<Point> boxes;
    public final int g;


    // Creates a new State
    public State (Point p, Set<Point> b, int g) {

        this.player = p;
        this.boxes = Collections.unmodifiableSet(b);
        this.g = g;
    }


    // Checks if all boxes are on goal cells
    public boolean isGoal(Board b) {

        for (Point x : boxes)
            if(!b.goals[x.r][x.c])
                return false;
        return true;
    }


    // Hash code depends only on boxes
    @Override public int hashCode() {
        return boxes.hashCode() * 31 + player.hashCode();
    }


    // Checks if two states are equal based on the boxes
    @Override public boolean equals(Object o) {
        if(!(o instanceof State)) return false;
        State s = (State)o;
        return player.equals(s.player) && boxes.equals(s.boxes);
    }


    @Override 
    public String toString() {
        return "State(player=" + player + ", boxes= " + boxes + ", g=" + g + ")";
    }
}
