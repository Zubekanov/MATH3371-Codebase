import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.cmatrix import *
import q1
import random

def test_householder():
    # Generating a random column vector with length between 1 and 10
    # and elements between -10 and 10
    rand_len = random.randint(1, 10)
    elements = [random.randint(-10, 10) for _ in range(rand_len)]
    x = cMatrix(elements).T
    # k = random.randint(1, rand_len)
    k = 1
    v, tau = q1.get_householder(x, k)

    # First condition H^2 = I
    H = cMatrix.identity(x.rows) - tau * v * v.T
    product = H * H
    assert product == cMatrix.identity(x.rows), f"H^2 != I: {product} != {cMatrix.identity(x.rows)}"

    # Second condition x = H * y for y = H * x
    y = H * x
    assert y == x, f"x != H * y: {x} != {y}"

    # Third condition ||x|| = ||y||
    assert x.norm() == y.norm(), f"||x|| != ||y||: {x.norm()} != {y.norm()}"

    # Fourth condition y_k+1:m = 0
    for i in range(k, x.rows):
        assert y[i][0] == 0, f"y[{i}] != 0: {y[i][0]} != 0"

    print("All tests passed!")

if __name__ == "__main__":
    test_householder()
