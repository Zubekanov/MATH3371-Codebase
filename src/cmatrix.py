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
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        det = 0
        for c in range(n):
            minor = [row[:c] + row[c+1:] for row in matrix[1:]]
            det += ((-1) ** c) * matrix[0][c] * self._calculate_determinant(minor)
        return det

    @property
    def cholesky(self):
        if _cache_keys.cholesky.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.cholesky.value] = self._cholesky_decomposition()
        return self._cached_values[_cache_keys.cholesky.value]
    
    # A = R^T * R, with R upper triangular.
    def _cholesky_decomposition(self):
        n = self.rows
        for i in range(n):
            for j in range(i):
                if self[i][j] != self[j][i]:
                    raise ValueError("Matrix must be symmetric")
        R = self.copy()
        for i in range(n):
            s = R[i][i] - sum(R[k][i] * R[k][i] for k in range(i))
            if s <= 0:
                raise ValueError("Matrix is not positive definite")
            R[i][i] = sqrt(s)
            for j in range(i+1, n):
                s_off = R[i][j] - sum(R[k][i] * R[k][j] for k in range(i))
                R[i][j] = s_off / R[i][i]
                R[j][i] = sympify(0)
        return R.T

    @property
    def lu(self):
        if _cache_keys.lu.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.lu.value] = self._lu_decomposition()
        return self._cached_values[_cache_keys.lu.value]
    
    # A = LU, L is lower triangular, U is upper triangular.
    def _lu_decomposition(self):
        # Check if any of the pivots are zero or very close to zero.
        if any(abs(self[i][i]) < 1e-10 for i in range(self.rows)):
            # TODO: Implement pivoting.
            raise ValueError("Matrix is singular")
        result = self.copy()
        # Perform the factorisation in place.
        for i in range(result.rows):
            for j in range(i+1, result.rows):
                result[j][i] /= result[i][i]
                for k in range(i+1, result.cols):
                    result[j][k] -= result[j][i] * result[i][k]
        # Separate the L and U matrices.
        L = [[result[i][j] if i > j else 0 for j in range(result.cols)] for i in range(result.rows)]
        # Add 1 to the diagonal of L.
        for i in range(result.rows):
            L[i][i] = 1
        U = [[result[i][j] if i <= j else 0 for j in range(result.cols)] for i in range(result.rows)]
        return (cMatrix(L), cMatrix(U))
    
    @property
    def qr(self):
        if _cache_keys.qr.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.qr.value] = self._qr_decomposition()
        return self._cached_values[_cache_keys.qr.value]
    
    # Performs QR decomposition using Householder transformations.
    # A = QR, Q is orthogonal, R is upper triangular.
    def _qr_decomposition(self):
        """
        Performs QR decomposition using Householder reflections.
        Returns a tuple (Q, R) where Q is orthogonal and R is upper triangular.
        """
        m = self.rows
        n = self.cols
        A_work = self.copy()
        Q_accumulated = cMatrix.identity(m)

        for k in range(min(m, n)):
            x_list = [[A_work[i][k]] for i in range(k, m)]
            x = cMatrix(x_list)

            norm_x = sqrt(sum(float(x[i][0])**2 for i in range(x.rows)))
            if norm_x == 0:
                H_k = cMatrix.identity(x.rows)
            else:
                s = 1 if x[0][0] >= 0 else -1
                alpha = -s * norm_x
                e1 = cMatrix([[1]] + [[0]] * (x.rows - 1))
                u = x - (alpha * e1)
                norm_u = sqrt(sum(float(u[i][0])**2 for i in range(u.rows)))
                if norm_u == 0:
                    v = u
                else:
                    v = (1 / norm_u) * u
                vvt = v * v.T
                H_k = cMatrix.identity(x.rows) - (2 * vvt)

            H_full = cMatrix.identity(m)
            for i in range(k, m):
                for j in range(k, m):
                    H_full[i][j] = H_k[i - k][j - k]

            A_work = H_full * A_work
            Q_accumulated = Q_accumulated * H_full

        return (Q_accumulated, A_work)

    def norm(self, p: int = 1, inf = False):
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

    # Ax = b -> Rx = Q^Tb
    def least_squares(self, b):
        if not b.rows == self.rows:
            raise ValueError("Incorrect dimensions for least squares. Matrix rows must match vector length.")
        lhs = self.qr[1]
        rhs = self.qr[0].T * b
        x = [0] * self.cols
        # Back substitution
        for i in reversed(range(self.cols)):
            x[i] = (rhs[i] - sum(lhs[i][j] * x[j] for j in range(i+1, self.cols))) / lhs[i][i]
        return cMatrix([x])

    @property
    def eigenpairs(self):
        pass

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
    test_matrix = cMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(test_matrix)
    test_imaginary_matrix = cMatrix([[1, 2 + 3*i, 3], [4, 5, 6], [7, 8, 9]])
    print(test_imaginary_matrix)
    test_imaginary_multiplication = (i * pi * test_imaginary_matrix)
    print(test_imaginary_multiplication)