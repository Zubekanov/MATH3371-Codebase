from enum import Enum

class _cache_keys(Enum):
    cols        = "columns"
    rows        = "rows"
    det         = "determinant"
    transpose   = "transpose"

class cMatrix:
    # Cache for calculatable properties.
    _cached_values = {}
    # Decorator to reset cache
    def _reset_cache(func):
        def wrapper(self, *args, **kwargs):
            self._cached_values = {}
            return func(self, *args, **kwargs)
        return wrapper
    
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

    def lu_decomposition(self):
        if self.rows != self.cols:
            raise ValueError("LU decomposition requires a square matrix.")
        n = self.rows
        L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        U = [[0.0 for _ in range(n)] for _ in range(n)]
        multipliers = {}
        counter = 1
        
        for k in range(n):
            for j in range(k, n):
                U[k][j] = self._row_major_contents[k][j] - sum(L[k][s] * U[s][j] for s in range(k))
            for i in range(k + 1, n):
                L[i][k] = (self._row_major_contents[i][k] - sum(L[i][s] * U[s][k] for s in range(k))) / U[k][k]
                multipliers[f"m{counter}"] = L[i][k]
                counter += 1
        return L, U, multipliers

    def __init__(self, contents: list[list[float]]):
        self._row_major_contents = contents

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
        # Align columns
        max_widths = [max([len(str(row[i])) for row in self._row_major_contents]) for i in range(self.cols)]
        res = ""
        for row in self._row_major_contents:
            res += "|"
            for i in range(len(row)):
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
        if isinstance(value, (int, float)):
            return cMatrix([[self[i][j] + value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented

    def __radd__(self, value):
        return NotImplemented
    
    def __sub__(self, value):
        if isinstance(value, cMatrix):
            if value.cols != self.cols or value.rows != self.rows:
                raise ValueError(f"Incorrect dimensions for subtraction. {self.rows}x{self.cols} != {value.rows}x{value.cols}")
            return cMatrix([[self[i][j] - value[i][j] for j in range(self.cols)] for i in range(self.rows)])
        if isinstance(value, (int, float)):
            return cMatrix([[self[i][j] - value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rsub__(self, value):
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, cMatrix):
            if self.cols != value.rows:
                raise ValueError(f"Incorrect dimensions for multiplication. {self.rows}x{self.cols} cannot multiply {value.rows}x{value.cols}")
            return cMatrix([[sum(self[i][k] * value[k][j] for k in range(self.cols)) for j in range(value.cols)] for i in range(self.rows)])
        if isinstance(value, (int, float)):
            return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented
    
    def __rmul__(self, value):
        if isinstance(value, (int, float)):
            return cMatrix([[self[i][j] * value for j in range(self.cols)] for i in range(self.rows)])
        return NotImplemented

if __name__ == "__main__":
    test_matrix = cMatrix([[ 5, -1,  3,  0],
                           [ 2,  3,  0,  1],
                           [ 3,  0,  2,  4],
                           [ 1,  6, -7,  3]])
    print(f"Matrix:\n{test_matrix}")
    print(f"LU Decomposition:\n{test_matrix.lu_decomposition()}")
