import numpy as np
import collections


class coo_matrix:
    def __init__(self, arg1, shape=None) -> None:
        self._data = dict()
        data, (row, col) = arg1
        for d, i, j in zip(data, row, col):
            if d == 0.0:
                continue
            val = self._data.get((i, j), 0.0)
            self._data[(i, j)] = val + d
        self.data = list(self._data.values())
        self.row, self.col = [], []
        for x in self._data.keys():
            self.row.append(x[0])
            self.col.append(x[1])
        self.shape = shape
        self._validate()
        self._sort()

    def _validate(self):
        assert len(self.data) == len(self.row)
        assert len(self.data) == len(self.col)
        assert len(self.data) <= self.shape[0] * self.shape[1]
        assert 0 <= min(self.row) and max(self.row) < self.shape[0]
        assert 0 <= min(self.col) and max(self.col) < self.shape[1]

    def _sort(self):
        data, row, col = [], [], []
        for d, r, c in sorted(zip(self.data, self.row, self.col), key=lambda x: (x[1], x[2])):
            data.append(d)
            row.append(r)
            col.append(c)
        self.data, self.row, self.col = data, row, col

    def __str__(self) -> str:
        _str = ""
        for d, i, j in zip(self.data, self.row, self.col):
            _str += " ".join([str(d), str(i), str(j)])
            _str += "\n"
        return _str

    def tocsr(self):
        indices = self.col
        c = collections.Counter(self.row)
        indptr = [c.get(0, 0)]
        for i in range(1, self.shape[0]):
            count = c.get(i, 0)
            indptr.append(indptr[i - 1] + count)
        return csr_matrix((self.data, indices, indptr), shape=self.shape)

    def tocsc(self):
        indices = self.row
        c = collections.Counter(self.col)
        indptr = [c.get(0, 0)]
        for i in range(1, self.shape[1]):
            count = c.get(i, 0)
            indptr.append(indptr[i - 1] + count)
        return csc_matrix((self.data, indices, indptr), shape=self.shape)


class csr_matrix:
    def __init__(self, arg1, shape=None) -> None:
        data, indices, indptr = arg1
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def getrow(self, i):
        if i == 0:
            return self.data[:self.indptr[0]], self.indices[:self.indptr[0]]
        start = i - 1
        end = i
        data = self.data[self.indptr[start]:self.indptr[end]]
        col = self.indices[self.indptr[start]:self.indptr[end]]
        return data, col

    def diagonal(self):
        diag = []
        for i in range(self.shape[0]):
            data, col = self.getrow(i)
            if i in col:
                c_idx = col.index(i)
                diag.append(data[c_idx])
            else:
                diag.append(0.0)
        return diag

    def __matmul__(self, other):
        if type(other) is not list and type(other) is not np.ndarray and type(other) is not csr_matrix:
            raise TypeError()
        if type(other) is list or type(other) is np.ndarray:
            if type(other) is np.ndarray:
                # Faster random access
                other = other.tolist()
            out = np.zeros_like(other)
            for i in range(self.shape[0]):
                data, col = self.getrow(i)
                # No.1
                # rowdense = np.zeros(self.shape[1])
                # rowdense[col] = data
                # out[i] = (rowdense * other).sum()

                # No.2
                val = 0.0
                for d, c in zip(data, col):
                    val += d * other[c]
                out[i] = val

                # No.3
                # out[i] = (np.asarray(data) * other[col]).sum()
        if type(other) is csr_matrix:
            data, row, col = [], [], []
            other = other.tocoo().tocsc()
            for i in range(self.shape[0]):
                data1, col1 = self.getrow(i)
                for j in range(other.shape[1]):
                    data2, row2 = other.getcol(j)
                    data_sum = 0.0
                    nonzero = False
                    for r_idx2, r in enumerate(row2):
                        if r in col1:
                            c_idx1 = col1.index(r)
                            nonzero = True
                            data_sum += data1[c_idx1] * data2[r_idx2]
                            #print(data_sum, data1[c_idx1], data2[r_idx2])
                    if nonzero:
                        data.append(data_sum)
                        row.append(i)
                        col.append(j)
            out = coo_matrix((data, (row, col)), shape=(self.shape[0], other.shape[1])).tocsr()
        return out

    def tocoo(self):
        data = self.data
        row = self.indices
        col = []
        for i in range(self.shape[0]):
            if i == 0:
                num = self.indptr[0]
            else:
                num = self.indptr[i] - self.indptr[i - 1]
            if num > 0:
                col += [i] * num
        return coo_matrix((data, (row, col)), shape=self.shape)


class csc_matrix:
    def __init__(self, arg1, shape=None) -> None:
        data, indices, indptr = arg1
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def getcol(self, i):
        if i == 0:
            return self.data[:self.indptr[0]], self.indices[:self.indptr[0]]
        start = i - 1
        end = i
        data = self.data[self.indptr[start]:self.indptr[end]]
        row = self.indices[self.indptr[start]:self.indptr[end]]
        return data, row
