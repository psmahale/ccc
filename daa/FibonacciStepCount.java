

import java.util.Scanner;

public class FibonacciStepCount {
    public static void main(String[] args) {
        // Step counter variable to measure the number of operations (steps)
        int stepCount = 0;

        // Scanner for user input
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the number of terms (n): ");
        stepCount++; // Step for displaying prompt

        int n = sc.nextInt(); // Read user input
        stepCount++; // Step for reading input

        // Validate input for negative numbers
        if (n < 0) {
            System.out.println("n must be >= 0");
            stepCount++; // Step for printing message

            sc.close();
            stepCount++; // Step for closing scanner

            System.out.println("Total Step Count: " + stepCount);
            return; // Exit program if invalid input
        }

        sc.close();
        stepCount++; // Step for closing scanner (for valid case)

        // Initialize first two Fibonacci numbers
        int first = 0;
        stepCount++; // Step for initializing first term

        int second = 1;
        stepCount++; // Step for initializing second term

        System.out.print("Fibonacci Series: ");
        stepCount++; // Step for printing label

        int i = 0;
        stepCount++; // Step for initializing loop counter

        // Loop to generate Fibonacci series up to n terms
        while (true) {
            stepCount++; // Step for comparison (i < n)

            if (!(i < n)) // Check loop condition
                break;

            // Print current Fibonacci term
            System.out.print(first + " ");
            stepCount++; // Step for printing term

            // Generate next Fibonacci number
            int next = first + second;
            stepCount++; // Step for addition + assignment

            first = second;
            stepCount++; // Step for updating first

            second = next;
            stepCount++; // Step for updating second

            i++;
            stepCount++; // Step for incrementing counter
        }

        stepCount++; // Step for final failed comparison

        System.out.println("\nTotal Step Count: " + stepCount);
        stepCount++; // Step for final print
    }
}

// Step Counting:

// The stepCount variable keeps track of each fundamental operation such as assignments, comparisons, prints, and input operations.

// This helps in analyzing algorithmic efficiency experimentally.