import java.util.Scanner;

public class BinomialCoefficientDP {

    // Function to calculate Binomial Coefficient (nCr) using DP
    public static int binomialCoeff(int n, int r) {
        int[][] C = new int[n + 1][r + 1];

        // Using Pascal's Triangle relation
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= Math.min(i, r); j++) {
                if (j == 0 || j == i)
                    C[i][j] = 1; // Base case: nC0 = nCn = 1
                else
                    C[i][j] = C[i - 1][j - 1] + C[i - 1][j]; // DP relation
            }
        }

        return C[n][r];
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter n: ");
        int n = sc.nextInt();

        System.out.print("Enter r: ");
        int r = sc.nextInt();

        int result = binomialCoeff(n, r);
        System.out.println("\nBinomial Coefficient C(" + n + ", " + r + ") = " + result);

        sc.close();
    }
}

/*
 * Enter n: 5
 * Enter r: 2
 * 
 */
