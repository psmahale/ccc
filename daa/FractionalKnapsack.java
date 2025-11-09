

import java.util.Arrays;
import java.util.Scanner;

class Item {
    int weight;      // Weight of the item
    int profit;      // Profit value of the item
    double ratio;    // Profit-to-weight ratio (used for sorting)

    // Constructor to initialize weight, profit, and ratio
    Item(int weight, int profit) {
        this.weight = weight;
        this.profit = profit;
        this.ratio = (double) profit / weight; // compute ratio for greedy sorting
    }
}

public class FractionalKnapsack {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Step 1: Input number of items
        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        Item[] items = new Item[n]; // Array to store items

        // Step 2: Input profit and weight for each item
        for (int i = 0; i < n; i++) {
            System.out.print("Enter profit and weight for item " + (i + 1) + ": ");
            int profit = sc.nextInt();
            int weight = sc.nextInt();
            items[i] = new Item(weight, profit); // create new item object
        }

        // Step 3: Input knapsack capacity
        System.out.print("Enter capacity of knapsack: ");
        int capacity = sc.nextInt();

        // Step 4: Sort items based on profit/weight ratio in descending order (Greedy step)
        Arrays.sort(items, (a, b) -> Double.compare(b.ratio, a.ratio));

        // Step 5: Select items greedily
        double totalProfit = 0.0;
        int remainingCapacity = capacity;

        for (int i = 0; i < n; i++) {
            // If the whole item can fit in the knapsack
            if (items[i].weight <= remainingCapacity) {
                totalProfit += items[i].profit;          // take the full item
                remainingCapacity -= items[i].weight;    // reduce available capacity
            } 
            // If only a fraction can fit
            else {
                totalProfit += items[i].ratio * remainingCapacity; // take fraction
                break; // knapsack is full
            }
        }

        // Step 6: Display result
        System.out.println("\nMaximum Profit = " + totalProfit);

        sc.close();
    }
}

/*
Sample Input/Output:
--------------------
Enter number of items: 3
Enter profit and weight for item 1: 60 10
Enter profit and weight for item 2: 100 20
Enter profit and weight for item 3: 120 30
Enter capacity of knapsack: 50

Output:
Maximum Profit = 240.0
*/
