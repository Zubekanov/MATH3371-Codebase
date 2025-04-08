import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.cmatrix import *
import sympy

def get_householder(x: cMatrix, k: int):
    print(f"Input vector x:\n{x}")
    print(f"Input k: {k}")

    # Check that x is a column vector
    if x.shape[1] != 1:
        raise ValueError("x must be a column vector")
    if k < 1 or k > x.rows:
        raise ValueError("k must be between 1 and the number of rows in x")
    
    v = x.copy()
    print(f"Copied vector v:\n{v}")

    # Shift k to zero indexing.
    k = k - 1
    print(f"Zero-indexed k: {k}")

    lower_norm = sympy.Rational(0)

    # Set the first k elements to 0 and calculate the norm of the rest.
    for i in range(v.rows):
        if i < k:
            v[i][0] = sympy.Rational(0)
        else:
            lower_norm += v[i][0] ** 2
        print(f"v after iteration {i}:\n{v}")
        print(f"lower_norm after iteration {i}: {lower_norm}")

    lower_norm = sympy.sqrt(lower_norm)
    print(f"Calculated lower_norm: {lower_norm}")

    # If the first element is negative, we need to flip the sign of the norm.
    if v[k][0] < 0:
        lower_norm = -lower_norm
    print(f"Adjusted lower_norm (if v[k][0] < 0): {lower_norm}")
        
    lower_norm = lower_norm + v[k][0]
    print(f"Updated lower_norm after adding v[k][0]: {lower_norm}")

    v[k][0] = sympy.Rational(1)
    print(f"v after setting v[k][0] to 1:\n{v}")

    for i in range(k + 1, v.rows):
        v[i][0] = v[i][0] / lower_norm
        print(f"v after normalizing element {i}:\n{v}")
    
    tau = sympy.Rational(2) / (v.norm() ** 2)
    print(f"Calculated tau: {tau}")

    return v, tau
