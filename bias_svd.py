import numpy as np

if __name__ == '__main__':
    R = np.mat([[1, 1, 1],
                [10, 10, 10],
                [5, 1, 1],
                [7, 10, 10],
                [5, 1, np.nan],
                [7, 10, np.nan],
                [5, 1, np.nan],
                [7, 10, np.nan],
                [5, 1, np.nan],
                [7, 10, np.nan],
                [5, 1, np.nan],
                [7, 10, np.nan]])

    # Robust covariance
    R_masked = np.ma.masked_invalid(R)
    print(np.ma.cov(R_masked, rowvar=False, allow_masked=True))

    # Replace missing values by the mean of the column
    mean = np.nanmean(R[:, 2])
    inds = np.where(np.isnan(R))
    R[inds] = mean

    print(np.cov(R, rowvar=False))
