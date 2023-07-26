import numpy as np
from abc import ABC, abstractmethod
import time

class FlatteningAlgo(ABC):
    @abstractmethod  
    def flatten(self, X):
        pass
    @abstractmethod
    def unflatten(self, x):
        pass

class WindowFlatten(FlatteningAlgo):
    def flatten(self, X, w=2):
        self._w = w
        M = X.shape[0]
        x = np.zeros(M*M)
        start = 0
        block_length = w*w
        end = block_length
        for i in range(0, M, w):
            for j in range(0, M, w):
                x[start:end] = X[i:i+w, j:j+w].flatten()
                start = end
                end += block_length
        return x
    
    def unflatten(self, x):
        M = int(np.sqrt(x.shape[0]))
        w = self._w
        X = np.zeros((M, M))
        start = 0
        block_length = w*w
        end = block_length
        for i in range(0, M, w):
            for j in range(0, M, w):
                X[i:i+w, j:j+w] = x[start:end].reshape((w, w))
                start = end
                end += block_length
        return X
    


class WindowContiguousFlatten(FlatteningAlgo):
    def flatten(self, X, w=2):
        self._w = w
        i, j = 0, 0
        M = X.shape[0]
        x = np.zeros(M*M)
        start = 0
        block_length = w*w
        end = block_length
        on_edge_cnt = 0
        reverse = False
        while end <= M*M:
            x[start:end] = X[i:i+w, j:j+w].flatten()
            if j == M-w or (j == 0 and i != 0):
                on_edge_cnt += 1
            if on_edge_cnt == 1:
                i+=w
            elif on_edge_cnt == 2:
                reverse = not reverse
                on_edge_cnt = 0 
                if reverse:
                    j-=w
                else:
                    j+=w
            elif not reverse:
                j+=w
            else:
                j-=w
            start = end
            end += block_length
        return x

        # flatten the matrix such that 
        
    def unflatten(self, x):
        M = int(np.sqrt(x.shape[0]))
        w = self._w
        X = np.zeros((M, M))
        start = 0
        block_length = w*w
        end = block_length
        i, j = 0, 0
        on_edge_cnt = 0
        reverse = False
        while end <= M*M:
            X[i:i+w, j:j+w] = x[start:end].reshape((w, w))
            if j == M-w or (j == 0 and i != 0):
                on_edge_cnt += 1
            if on_edge_cnt == 1:
                i+=w
            elif on_edge_cnt == 2:
                reverse = not reverse
                on_edge_cnt = 0 
                if reverse:
                    j-=w
                else:
                    j+=w
            elif not reverse:
                j+=w
            else:
                j-=w
            start = end
            end += block_length
        return X

class BaseFlatten(FlatteningAlgo):
    def flatten(self, X, order='C'):
        self._order = order
        return X.flatten(order)
    def unflatten(self, x):
        return x.reshape(int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])), order=self._order)
     
# correctness and runtime tests
if __name__ == "__main__":
    # X = np.array([[1, 1, 2, 2, 3, 3, 4, 4],
    #               [1, 1, 2, 2, 3, 3, 4, 4],
    #               [5, 5, 6, 6, 7, 7, 8, 8],
    #               [5, 5, 6, 6, 7, 7, 8, 8],
    #               [9, 9, 10, 10, 11, 11, 12, 12],
    #                 [9, 9, 10, 10, 11, 11, 12, 12],
    #                 [13, 13, 14, 14, 15, 15, 16, 16],
    #                 [13, 13, 14, 14, 15, 15, 16, 16]
    #               ])
    # X = np.array([[1, 1, 2, 2],
    #               [1, 1, 2, 2],
    #               [4, 4, 3, 3],
    #               [4, 4, 3, 3]])

    # test the two algorithms and time them


    X = np.random.normal(size=(5000, 5000))

    spatial_flatten = WindowContiguousFlatten()
    start = time.time()
    flattened = spatial_flatten.flatten(X)
    X_reconstructed = spatial_flatten.unflatten(flattened)
    end = time.time()
    assert np.allclose(X, X_reconstructed)

    print("WindowContiguousFlatten took {} seconds".format(end-start))

    spatial_flatten = WindowFlatten()
    start = time.time()
    flattened = spatial_flatten.flatten(X)
    X_reconstructed = spatial_flatten.unflatten(flattened)
    end = time.time()
    assert np.allclose(X, X_reconstructed)

    print("WindowFlatten took {} seconds".format(end-start))

    spatial_flatten = BaseFlatten()
    start = time.time()
    flattened = spatial_flatten.flatten(X)
    X_reconstructed = spatial_flatten.unflatten(flattened)
    end = time.time()
    assert np.allclose(X, X_reconstructed)

    print("BaseFlatten took {} seconds".format(end-start))
