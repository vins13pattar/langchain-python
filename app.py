def calculator(a, op, b):
    """Performs basic arithmetic operations."""
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            return "Error: Division by zero"
        return a / b
    else:
        return "Error: Invalid operation"

if __name__ == '__main__':
    # Simple demonstration (will be tested separately)
    print(f"5 + 3 = {calculator(5, '+', 3)}")
    print(f"10 / 2 = {calculator(10, '/', 2)}")