/**
 * Class used to represent grid coordinates 
 * on the Sokoban board (stores: row index r, column index c) 
 * 
*/

public class Point {

    public final int r, c;

    public Point(int r, int c) {
        this.r = r;
        this.c = c;
    }


    // Returns a new Point moved by (dr,dc)
    public Point move(int dr, int dc) {
        return new Point(r + dr, c + dc);
    }


    // Hash code based on row and column (required for HashSet/HashMap)
    @Override public int hashCode() {
        return r * 31 + c;
    }


    @Override public boolean equals(Object o) {
        if (!(o instanceof Point)) return false;
        Point p = (Point) o;
        return r == p.r && c == p.c;
    }
    

    @Override public String toString() {
        return "(" + r + "," + c + ")" ;
    }

}
