# Import library
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return (x + 3)**2

def df(x):
    return 2 * (x + 3)

# Initialize parameters
x = 2                # starting point
learning_rate = 0.1  # step size
iterations = 25      # number of steps
x_values = []        # store x for plotting
y_values = []        # store y for plotting

# Perform Gradient Descent
for i in range(iterations):
    grad = df(x)                 # compute gradient
    x = x - learning_rate * grad # update x
    y = f(x)
    x_values.append(x)
    y_values.append(y)
    print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {y:.4f}")

# Final results
x_min = round(x, 4)
y_min = round(f(x_min), 4)

print("\nLocal minima occurs at:")
print(f"x = {x_min}")
print(f"Minimum value of y = {y_min}")

# Visualization
plt.plot(x_values, y_values, 'bo-', label='Descent Path')
plt.title('Gradient Descent on y = (x + 3)²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
# 🧮 GRADIENT DESCENT ALGORITHM — EXPLANATION 🧮
# ----------------------------------------------
# This code demonstrates how **Gradient Descent** works on a simple quadratic function y = (x + 3)².
#
# 1️⃣ **Function Definition:**
# The function f(x) = (x + 3)² has a minimum point at x = -3.  
# Its derivative df(x) = 2(x + 3) gives the slope (gradient) at each point.
#
# 2️⃣ **Initialization:**
# We start from an initial value x = 2 (a random guess), set a learning rate of 0.1 to control step size,  
# and plan to take 25 iterations to approach the minimum.
#
# 3️⃣ **Gradient Descent Loop:**
# In each iteration:
#   - Compute the gradient (df(x)) → tells us the slope direction.
#   - Update x using the formula: x_new = x_old - learning_rate * gradient  
#     This moves x in the opposite direction of the slope (towards the minimum).
#   - Evaluate f(x) after the update and store results for visualization.
#
# 4️⃣ **Result:**
# After several iterations, x converges close to -3, where f(x) is minimum (≈ 0).
#
# 5️⃣ **Visualization:**
# The graph shows the path of gradient descent (blue points) — how x gradually moves towards the minimum point.
#
# 💡 Conceptually, this demonstrates how optimization algorithms iteratively reduce error/loss 
# by following the negative gradient direction until reaching the lowest value (local minima).
