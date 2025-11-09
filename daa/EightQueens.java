import java.util.Scanner;

public class EightQueens {

    static final int N = 8;
    static int[][] board = new int[N][N];

    // Function to print the 8-Queens matrix
    public static void printBoard() {
        System.out.println("\nFinal 8-Queens Solution:");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }

    // Function to check if a queen can be placed safely
    static boolean isSafe(int row, int col) {
        int i, j;

        // Check this row on the left side
        for (i = 0; i < col; i++)
            if (board[row][i] == 1)
                return false;

        // Check upper diagonal on the left side
        for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1)
                return false;

        // Check lower diagonal on the left side
        for (i = row, j = col; i < N && j >= 0; i++, j--)
            if (board[i][j] == 1)
                return false;

        return true;
    }

    // Recursive backtracking function
    static boolean solveNQ(int col) {
        // If all queens are placed
        if (col >= N)
            return true;

        // Try placing this queen in all rows one by one
        for (int i = 0; i < N; i++) {
            if (isSafe(i, col)) {
                board[i][col] = 1; // Place queen

                // Recur to place rest of the queens
                if (solveNQ(col + 1))
                    return true;

                // If placing queen here leads to no solution
                board[i][col] = 0; // Backtrack
            }
        }

        return false;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the position of the first queen (row and column, 0-7): ");
        int row = sc.nextInt();
        int col = sc.nextInt();
        sc.close();

        // Place first queen manually
        board[row][col] = 1;

        // Start solving from next column
        if (solveNQ(col + 1))
            printBoard();
        else
            System.out.println("No solution exists.");
    }
}

// Algorithm Used — Backtracking

// Place a queen column by column.

// For each column, check every row using the isSafe() function:

// Checks left row, upper-left diagonal, and lower-left diagonal for conflicts.

// If safe → place queen and move to next column.

// If no safe row → backtrack (remove previous queen and try next position).

// Continue until all 8 queens are placed.

// Mathematical / Numerical Idea

// For an 8×8 board →

// We must find 1 queen per column → total 8 columns.

// Total possible placements without constraints = 
// 8
// 8
// =
// 16
// ,
// 777
// ,
// 216
// 8
// 8
// =16,777,216.

// Backtracking prunes invalid moves → explores only a small subset efficiently.

// Final number of distinct solutions = 92 (standard known result).