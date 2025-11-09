
import java.util.PriorityQueue;
import java.util.Scanner;

// Node class representing each character in Huffman Tree
class HuffmanNode {
    int data;       // frequency of character
    char c;         // character itself
    HuffmanNode left, right; // left and right child nodes

    // Constructor to initialize data members
    HuffmanNode(char c, int data) {
        this.c = c;
        this.data = data;
        left = right = null;
    }
}

// Comparator class to sort HuffmanNodes by frequency (ascending order)
class MyComparator implements java.util.Comparator<HuffmanNode> {
    public int compare(HuffmanNode x, HuffmanNode y) {
        return x.data - y.data; // smaller frequency comes first
    }
}

public class HuffmanEncodingGreedy {
    static int stepCount = 0; // to count total steps for DAA analysis

    // Recursive function to print Huffman Codes for each character
    public static void printCode(HuffmanNode root, String s) {
        stepCount++; // count recursive call step

        // Base case: if leaf node is reached, print the character and its code
        if (root.left == null && root.right == null && Character.isLetter(root.c)) {
            System.out.println(root.c + " : " + s);
            stepCount++; // step for printing
            return;
        }

        // Recurse left and right with added bits '0' and '1'
        printCode(root.left, s + "0");
        printCode(root.right, s + "1");
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Step 1: Input number of characters
        System.out.print("Enter number of characters: ");
        int n = sc.nextInt();
        stepCount++;

        char[] charArray = new char[n];
        int[] charFreq = new int[n];
        stepCount += 2;

        // Step 2: Input each character and its frequency
        for (int i = 0; i < n; i++) {
            System.out.print("Enter character: ");
            charArray[i] = sc.next().charAt(0);
            System.out.print("Enter frequency: ");
            charFreq[i] = sc.nextInt();
            stepCount += 2;
        }
        sc.close();

        // Step 3: Create a priority queue (min-heap) for building Huffman Tree
        PriorityQueue<HuffmanNode> q = new PriorityQueue<>(n, new MyComparator());
        stepCount++;

        // Step 4: Add all character nodes to the priority queue
        for (int i = 0; i < n; i++) {
            HuffmanNode node = new HuffmanNode(charArray[i], charFreq[i]);
            q.add(node);
            stepCount += 2;
        }

        HuffmanNode root = null; // root of Huffman Tree

        // Step 5: Build Huffman Tree using greedy approach
        while (q.size() > 1) {
            // Remove two nodes with the smallest frequency
            HuffmanNode x = q.poll();
            HuffmanNode y = q.poll();
            stepCount += 2;

            // Create a new internal node with frequency equal to sum of two smallest nodes
            HuffmanNode f = new HuffmanNode('-', x.data + y.data);
            f.left = x;
            f.right = y;
            root = f;
            stepCount += 3;

            // Add the new combined node back into the priority queue
            q.add(f);
            stepCount++;
        }

        // Step 6: Print Huffman Codes for each character
        System.out.println("\nHuffman Codes are:");
        printCode(root, "");

        // Step 7: Display total step count
        System.out.println("\nTotal Step Count: " + stepCount);
    }
}

/*
Sample Input/Output:
--------------------
Enter number of characters: 6
Enter character: a
Enter frequency: 5
Enter character: b
Enter frequency: 9
Enter character: c
Enter frequency: 12
Enter character: d
Enter frequency: 13
Enter character: e
Enter frequency: 16
Enter character: f
Enter frequency: 45

Output:
Huffman Codes are:
f : 0
c : 100
d : 101
a : 1100
b : 1101
e : 111

Total Step Count: 78
*/


// Huffman encoding is a lossless data compression algorithm.

// It assigns shorter binary codes to more frequent characters and longer codes to less frequent ones.

// The algorithm builds a binary tree (Huffman Tree) where each leaf node represents a character, and the path from root to leaf gives its binary code.

// Algorithm Steps:

// Create a node for each character and insert it into a min-priority queue (sorted by frequency).

// While there’s more than one node in the queue:

// Remove two nodes with the smallest frequencies.

// Create a new node with their combined frequency.

// Make the two nodes children of this new node.

// Insert the new node back into the queue.

// The remaining node is the root of the Huffman Tree.

// Traverse the tree to assign:

// 0 for left edges

// 1 for right edges

// The resulting binary string is the Huffman code.

// Example:
// Character	Frequency	Code
// f	45	0
// c	12	100
// d	13	101
// a	5	1100
// b	9	1101
// e	16	111
// Time Complexity:

// Building the Huffman Tree: O(n log n) (due to priority queue operations)

// Generating codes: O(n)

// Total Complexity: O(n log n)

// Space Complexity:

// O(n) for storing nodes and character arrays.

// Conclusion:

// The Huffman Encoding algorithm efficiently compresses data by generating variable-length prefix codes based on frequency, ensuring no ambiguity in decoding and optimal average code length.

// Mathematical Explanation (Short Form)

// Huffman Encoding minimizes the total weighted path length of all symbols.
// For each character 
// 𝑖
// i:

// Frequency = 
// 𝑓
// 𝑖
// f
// i
// 	​


// Code length = 
// 𝐿
// 𝑖
// L
// i
// 	​


// The goal is to minimize

// ∑
// 𝑓
// 𝑖
// ×
// 𝐿
// 𝑖
// ∑f
// i
// 	​

// ×L
// i
// 	​


// so that frequently used characters get shorter codes.

// Example:
// Given frequencies:
// a=5, b=9, c=12, d=13, e=16, f=45

// After building the Huffman tree:

// Char	fᵢ	Code	Lᵢ	fᵢ×Lᵢ
// f	45	0	1	45
// c	12	100	3	36
// d	13	101	3	39
// a	5	1100	4	20
// b	9	1101	4	36
// e	16	111	3	48
// Total
// =
// 224
// ,
// Sum of frequencies
// =
// 100
// Total=224,Sum of frequencies=100
// Average code length
// =
// 224
// /
// 100
// =
// 2.24
//  bits per symbol
// Average code length=224/100=2.24 bits per symbol