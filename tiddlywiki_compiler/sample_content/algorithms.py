# Sample Algorithm
def factorial(n):
    """
    Compute factorial using recursive algorithm
    Time complexity: O(n)
    Space complexity: O(n) due to recursion
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """
    Compute Fibonacci number
    Demonstrates exponential growth pattern
    """
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(f"5! = {factorial(5)}")
print(f"F(10) = {fibonacci(10)}")
