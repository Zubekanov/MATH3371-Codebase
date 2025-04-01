from enum import Enum
import sympy.functions.elementary.complexes as c
from sympy import I as i, E as e, pi, sympify, sqrt

class _cache_keys(Enum):
    cols        = "columns"
    rows        = "rows"
    det         = "determinant"
    transpose   = "transpose"
    lu          = "lu_decomposition"
    qr          = "qr_decomposition"
    inverse     = "inverse"
    householder = "householder_transformation"
    cholesky    = "cholesky_decomposition"

class cMatrix:
    # Decorator to reset cache
    def _reset_cache(func):
        def wrapper(self, *args, **kwargs):
            self._cached_values = {}
            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def identity(size: int):
        return cMatrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    # Finds the matrix satisfying G*lhs = rhs
    @staticmethod
    def solve_givens(lhs, rhs, p=None, q=None):
        n = len(lhs)
        if not p:
            for i in range(n):
                if lhs[i] != rhs[i]:
                    p = i
                    break
            if not p:
                raise ValueError("No entries differ in lhs and rhs")
        if not q:
            for i in range(p + 1, n):
                if lhs[i] != rhs[i]:
                    q = i
                    break
            if not q:
                raise ValueError("Only one entries differ in lhs and rhs")
        if len(rhs) != n:
            raise ValueError("lhs and rhs must be the same dimension.")
        if not (0 <= p < q < n):
            raise ValueError("Require 0 <= p < q < n.")
        
        for i in range(n):
            if i not in (p,q) and lhs[i] != rhs[i]:
                raise ValueError(f"lhs and rhs differ in coordinate {i}, but only {p,q} are allowed to differ.")
        
        a, b   = lhs[p], lhs[q]
        a_hat, b_hat = rhs[p], rhs[q]

        lhs_len_sq = a*a + b*b
        rhs_len_sq = a_hat*a_hat + b_hat*b_hat
        if abs(lhs_len_sq - rhs_len_sq) > 1e-12:
            raise ValueError("Rotation cannot change the 2D length.  Norms differ in the plane (p,q).")
        
        if lhs_len_sq == 0:
            return cMatrix.identity(n)
        
        denom = sympify(lhs_len_sq)
        dot   = a*a_hat + b*b_hat
        cross = a*b_hat - b*a_hat
        
        c_val = dot   / denom
        s_val = cross / denom
        
        length_cs = c_val*c_val + s_val*s_val
        if abs(length_cs - 1.0) > 1e-10:
            raise ValueError(f"Did not find a valid rotation, c^2 + s^2 = {length_cs}, expected 1.")
        
        G = cMatrix.identity(n)
        G[p][p] = c_val
        G[p][q] = s_val
        G[q][p] = -s_val
        G[q][q] = c_val
        
        return G
    
    @property
    def T(self):
        if _cache_keys.transpose.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.transpose.value] = [[self[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return cMatrix(self._cached_values[_cache_keys.transpose.value])
    
    @property
    def cols(self):
        if _cache_keys.cols.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.cols.value] = len(self._row_major_contents[0])
        return self._cached_values[_cache_keys.cols.value]
    
    @property
    def rows(self):
        if _cache_keys.rows.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.rows.value] = len(self._row_major_contents)
        return self._cached_values[_cache_keys.rows.value]
    
    @property
    def det(self):
        if _cache_keys.det.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.det.value] = self._calculate_determinant(self._row_major_contents)
        return self._cached_values[_cache_keys.det.value]
    
    def _calculate_determinant(self, matrix):
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise ValueError("Matrix must be square")

        if n == 1:
            return matrix[0][0].simplify()
        if n == 2:
            return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]).simplify()

        det = 0
        for c in range(n):
            minor = [row[:c] + row[c+1:] for row in matrix[1:]]
            det += ((-1) ** c) * matrix[0][c] * self._calculate_determinant(minor)
            
        return det.simplify()

    def norm(self, p: int = 2, inf = False):
        if inf:
            if "inf_norm" not in self._cached_values.keys():
                self._cached_values["inf_norm"] = self._calculate_norm(p, inf)
            return self._cached_values["inf_norm"]
        else:
            if f"{p}_norm" not in self._cached_values.keys():
                self._cached_values[f"{p}_norm"] = self._calculate_norm(p, inf)
            return self._cached_values[f"{p}_norm"]

    def _calculate_norm(self, p: int = 1, inf = False):
        if inf:
            return max(max(abs(cell) for cell in row) for row in self._row_major_contents)
        else:
            return (sum(sum(abs(cell) ** p for cell in row) for row in self._row_major_contents)) ** (1/p)

    @property
    def inverse(self):
        if _cache_keys.inverse.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.inverse.value] = self._calculate_inverse()
        return self._cached_values[_cache_keys.inverse.value]
    
    def _calculate_inverse(self):
        # Generate cofactors matrix.
        cofactors = [[((-1) ** (i+j)) * self._calculate_determinant([row[:j] + row[j+1:] for row in self._row_major_contents[:i] + self._row_major_contents[i+1:]]) for j in range(self.cols)] for i in range(self.rows)]
        adjugate = cMatrix(cofactors).T
        return adjugate * (1 / self.det)

    def simplify(self):
        self._row_major_contents = [[cell.simplify() for cell in row] for row in self._row_major_contents]

    def copy(self):
        copy = cMatrix(self._row_major_contents)
        copy._cached_values = self._cached_values.copy()
        return copy

    def __eq__(self, value):
        if not isinstance(value, cMatrix):
            return False
        return self._row_major_contents == value._row_major_contents

    def __getitem__(self, key):
        return self._row_major_contents[key]

    @_reset_cache
    def __setitem__(self, key, value):
        self._row_major_contents[key] = value

    def __str__(self):
        # Align columns and decimal points.
        max_widths = [max([len(f"{row[i]}") for row in self._row_major_contents]) for i in range(self.cols)]
        res = "\n"
        for row in self._row_major_contents:
            res += "|"
            for i in range(len(row)):
                    res += f"{str(row[i]):>{max_widths[i]}} "
            res += "|\n"
        return res

    def __repr__(self):
        return self.__str__()

    def __add__(self, value):
        if isinstance(value, cMatrix):
            if value.cols != self.cols or value.rows != self.rows:
                raise ValueError(f"Incorrect dimensions for addition. {self.rows}x{self.cols} != {value.rows}x{value.cols}")
            return cMatrix([[self[i][j] + value[i][j] for j in range(self.cols)] for i in range(self.rows)])
        
        return cMatrix([[self[i][j] + value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented

    def __radd__(self, value):
        return NotImplemented
    
    def __sub__(self, value):
        if isinstance(value, cMatrix):
            if value.cols != self.cols or value.rows != self.rows:
                raise ValueError(f"Incorrect dimensions for subtraction. {self.rows}x{self.cols} != {value.rows}x{value.cols}")
            return cMatrix([[self[i][j] - value[i][j] for j in range(self.cols)] for i in range(self.rows)])
        
        return cMatrix([[self[i][j] - value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rsub__(self, value):
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, cMatrix):
            if self.cols != value.rows:
                raise ValueError(f"Incorrect dimensions for multiplication. {self.rows}x{self.cols} cannot multiply {value.rows}x{value.cols}")
            return cMatrix([[sum(self[i][k] * value[k][j] for k in range(self.cols)) for j in range(value.cols)] for i in range(self.rows)])

        return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rmul__(self, value):
        return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __init__(self, contents: list[list]):
        if not all(len(row) == len(contents[0]) for row in contents):
            raise ValueError("Matrix must be rectangular")
        self._row_major_contents = [[sympify(cell).simplify() for cell in row] for row in contents]
        self._cached_values = {}



if __name__ == "__main__":
    print(cMatrix.solve_givens([7, -1, 3, 2, 4], [7, -1, 5, 2, 0]))