from enum import Enum
from fractions import Fraction

class _cache_keys(Enum):
    cols        = "columns"
    rows        = "rows"
    det         = "determinant"
    transpose   = "transpose"
    lu          = "lu_decomposition"
    qr          = "qr_decomposition"
    inverse     = "inverse"
    householder = "householder_transformation"

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
    def lu(self):
        if _cache_keys.lu.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.lu.value] = self.lu_decomposition()
        return self._cached_values[_cache_keys.lu.value]
    
    def lu_decomposition(self):
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
            self._cached_values[_cache_keys.qr.value] = self.qr_decomposition()
        return self._cached_values[_cache_keys.qr.value]
    
    def qr_decomposition(self):
        return NotImplemented

    @property
    def householder(self):
        if _cache_keys.householder.value not in self._cached_values.keys():
            self._cached_values[_cache_keys.householder.value] = self.householder_transformation()
        return self._cached_values[_cache_keys.householder.value]
    
    def householder_transformation(self):
        householder = cMatrix.identity(self.rows) - 2 * self * self.T / (self.T * self)
        return householder

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

    def __init__(self, contents: list[list]):
        if not all(len(row) == len(contents[0]) for row in contents):
            raise ValueError("Matrix must be rectangular")
        # Ints and Floats are converted to Fractions, Fractions are kept.
        if not all(isinstance(cell, (int, float, Fraction)) for row in contents for cell in row):
            raise ValueError("Matrix must contain only ints, floats, or Fractions")
        self._row_major_contents = [[Fraction(cell).limit_denominator() if isinstance(cell, (int, float)) else cell for cell in row] for row in contents]
        self._cached_values = {}

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
        max_widths = [max([len(f"{row[i]:.4f}" if isinstance(row[i], float) else str(row[i])) for row in self._row_major_contents]) for i in range(self.cols)]
        res = "\n"
        for row in self._row_major_contents:
            res += "|"
            for i in range(len(row)):
                if isinstance(row[i], float):
                    res += f"{row[i]:>{max_widths[i]}.4f} "
                else:
                    res += f"{str(row[i]):>{max_widths[i]}} "
            res += "|\n"
        return res

    def __repr__(self):
        return self.__str__()

    # TODO: Implement radd, rsub, rmul for numpy compatibility.
    def __add__(self, value):
        if isinstance(value, cMatrix):
            if value.cols != self.cols or value.rows != self.rows:
                raise ValueError(f"Incorrect dimensions for addition. {self.rows}x{self.cols} != {value.rows}x{value.cols}")
            return cMatrix([[self[i][j] + value[i][j] for j in range(self.cols)] for i in range(self.rows)])
        if isinstance(value, (int, float, Fraction)):
            return cMatrix([[self[i][j] + value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented

    def __radd__(self, value):
        return NotImplemented
    
    def __sub__(self, value):
        if isinstance(value, cMatrix):
            if value.cols != self.cols or value.rows != self.rows:
                raise ValueError(f"Incorrect dimensions for subtraction. {self.rows}x{self.cols} != {value.rows}x{value.cols}")
            return cMatrix([[self[i][j] - value[i][j] for j in range(self.cols)] for i in range(self.rows)])
        if isinstance(value, (int, float, Fraction)):
            return cMatrix([[self[i][j] - value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rsub__(self, value):
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, cMatrix):
            if self.cols != value.rows:
                raise ValueError(f"Incorrect dimensions for multiplication. {self.rows}x{self.cols} cannot multiply {value.rows}x{value.cols}")
            return cMatrix([[sum(self[i][k] * value[k][j] for k in range(self.cols)) for j in range(value.cols)] for i in range(self.rows)])
        if isinstance(value, (int, float, Fraction)):
            return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rmul__(self, value):
        if isinstance(value, (int, float, Fraction)):
            return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented

if __name__ == "__main__":
    test_matrix = cMatrix([[ 5, -1,  3,  0],
                           [ 2,  3,  0,  1],
                           [ 3,  0,  2,  4],
                           [ 1,  6, -7,  3]])
    print(f"Matrix:\n{test_matrix}")
    print(f"Inverse:\n{test_matrix.inverse}")
