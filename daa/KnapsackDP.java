import java.util.Scanner;

public class KnapsackDP {

    // Function to solve 0/1 Knapsack problem using Dynamic Programming
    public static int knapSack(int capacity, int[] weight, int[] profit, int n) {
        int[][] K = new int[n + 1][capacity + 1];

        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (i == 0 || w == 0)
                    K[i][w] = 0;
                else if (weight[i - 1] <= w)
                    K[i][w] = Math.max(profit[i - 1] + K[i - 1][w - weight[i - 1]], K[i - 1][w]);
                else
                    K[i][w] = K[i - 1][w];
            }
        }

        return K[n][capacity];
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        int[] profit = new int[n];
        int[] weight = new int[n];

        for (int i = 0; i < n; i++) {
            System.out.print("Enter profit of item " + (i + 1) + ": ");
            profit[i] = sc.nextInt();
            System.out.print("Enter weight of item " + (i + 1) + ": ");
            weight[i] = sc.nextInt();
        }

        System.out.print("Enter capacity of knapsack: ");
        int capacity = sc.nextInt();

        int maxProfit = knapSack(capacity, weight, profit, n);
        System.out.println("\nMaximum Profit = " + maxProfit);

        sc.close();
    }
}

/*
 * Enter number of items: 4
 * Enter profit of item 1: 10
 * Enter weight of item 1: 5
 * Enter profit of item 2: 40
 * Enter weight of item 2: 4
 * Enter profit of item 3: 30
 * Enter weight of item 3: 6
 * Enter profit of item 4: 50
 * Enter weight of item 4: 3
 * Enter capacity of knapsack: 10

 */
