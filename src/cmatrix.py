from enum import Enum

class _cache_keys(Enum):
    cols        = "columns"
    rows        = "rows"
    det         = "determinant"
    transpose   = "transpose"

class cMatrix:
    _row_major_contents = [[]]

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
        if _cache_keys.transpose.value not in self._cached_values.keys:
            self._cached_values[_cache_keys.transpose.value] = [[self[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return cMatrix(self._cached_values[_cache_keys.transpose.value])
    
    @property
    def cols(self):
        if _cache_keys.cols.value not in self._cached_values.keys:
            self._cached_values[_cache_keys.cols.value] = len(self._row_major_contents[0])
        return self._cached_values[_cache_keys.cols.value]
    
    @property
    def rows(self):
        if _cache_keys.rows.value not in self._cached_values.keys:
            self._cached_values[_cache_keys.rows.value] = len(self._row_major_contents)
        return self._cached_values[_cache_keys.rows.value]
    
    @property
    def det(self):
        if _cache_keys.det.value not in self._cached_values.keys:
            self._cached_values[_cache_keys.det.value] = self._calculate_determinant()
        return self._cached_values[_cache_keys.det.value]
    
    def _calculate_determinant(self):
        if self.cols != self.rows:
            raise ValueError("Matrix must be square to calculate determinant.")
        if self.cols == 1:
            return self[0][0]
        if self.cols == 2:
            return self[0][0] * self[1][1] - self[0][1] * self[1][0]
        det = 0
        for i in range(self.cols):
            det += self[0][i] * self._submatrix(0, i).det * (-1)**i
        return det
    
    def _submatrix(self, row, col):
        return cMatrix([row[:col] + row[col+1:] for row in self._row_major_contents[:row] + self._row_major_contents[row+1:]])

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
        return str(self._row_major_contents)

    def __repr__(self):
        return str(self._row_major_contents)

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
