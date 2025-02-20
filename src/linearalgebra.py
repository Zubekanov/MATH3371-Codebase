import datetime as dt
import numpy as np
import utildecorators as ud

class Lab01:
    @staticmethod
    def run():
        A = np.array([[ 2,  0, -1],
                      [ 5,  7,  3],
                      [-4,  1,  8],
                      [ 9,  4,  6]])
        
        B = np.array([[ 1,  1,  2],
                      [ 2, -1,  6],
                      [ 9,  5,  5]])
        
        x = np.array([[ 3],
                      [ 0],
                      [-5],
                      [ 2]])
        
        AB = A @ B
        print(f"A*B =\n{AB}\n")
        
        ATx = A.T @ x
        print(f"A^T*x =\n{ATx}\n")
        
        BAT = B @ A.T
        print(f"B*A^T =\n{BAT}\n")
        
        ATApB = A.T @ A + B
        print(f"A^T*A + B =\n{ATApB}\n")
        
        xTxm3 = x.T @ x - 3
        print(f"x^T*x - 3 = {xTxm3}\n")
        
    def timed_gemm(size):
        @ud.process_printer(f"gemm test of size {size}")
        def gemm_test(size):
            alpha = 3
            beta = -2
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            C = np.random.rand(size, size)
            res = alpha * A @ B + beta * C
            return res

        return gemm_test(size)
        
    def run_gemm_tests():
        # Run gemm tests increasing from 1 until time exceeds 30 seconds.
        size = 1
        while True:
            start = dt.datetime.now()
            res = Lab01.timed_gemm(size)
            end = dt.datetime.now()
            elapsed = end - start
            if elapsed.total_seconds() > 30:
                break
            size *= 2
        
if __name__ == "__main__":
    Lab01.run()
    Lab01.run_gemm_tests()
