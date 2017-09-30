import numpy as np

if __name__ == '__main__':
    R = np.mat([[1, 1, 1],
                [7, 7, 7],
                [3, 1, 1],
                [5, 7, 7],
                [3, 1, np.NA],
                [5, 7, np.NA],
                [3, 1, np.NA],
                [5, 7, np.NA],
                [3, 1, np.NA],
                [5, 7, np.NA],
                [3, 1, np.NA],
                [5, 4, np.NA]])
    
    sim = np.dot(R.transpose(), R)

    print(sim)
