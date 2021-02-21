import numpy as np
from cpd_nonlin import cpd_nonlin
from tools.cpd_auto import cpd_auto


def gen_data(n, m, d=1):
    """Generates data with change points
    n - number of samples
    m - number of change-points
    WARN: sigma is proportional to m
    Returns:
        X - data array (n X d)
        cps - change-points array, including 0 and n"""
    np.random.seed(1)
    # Select changes at some distance from the boundaries
    cps = np.random.permutation((n * 3 // 4) - 1)[0:m] + 1 + n // 8
    cps = np.sort(cps)
    cps = [0] + list(cps) + [n]
    mus = np.random.rand(m + 1, d) * (m / 2)  # make sigma = m/2
    X = np.zeros((n, d))
    for k in range(m + 1):
        X[cps[k]:cps[k + 1], :] = mus[k, :][np.newaxis, :] + np.random.rand(cps[k + 1] - cps[k], d)
    return (X, np.array(cps))


if __name__ == "__main__":


    # print("Test 1: 1-dimensional signal")
    # n = 1000
    # m = 10
    # (X, cps_gt) = gen_data(n, m)
    # print("Ground truth:", cps_gt)
    # K = np.dot(X, X.T)
    # cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
    # print("Estimated:", cps)
    # mi = np.min(X)
    # ma = np.max(X)
    # print("=" * 79)
    #
    # print("Test 2: multidimensional signal")
    # n = 1300
    # m = 50
    # (X, cps_gt) = gen_data(n, m, d=1024)
    # print("Ground truth:", cps_gt)
    # K = np.dot(X, X.T)
    # cps, scores = cpd_nonlin(K, m, lmin=1, lmax=10000)
    # print("Estimated:", cps)
    # mi = np.min(X)
    # ma = np.max(X)
    # print("=" * 79)

    print("Test 3: automatic selection of the number of change-points")
    n = 300
    m = 20
    (X, cps_gt) = gen_data(n, m, d=1024)
    print("Ground truth: (m=%d)" % m, cps_gt)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, 2 * m, 1)
    print("Estimated: (m=%d)" % len(cps), cps)
    mi = np.min(X)
    ma = np.max(X)
    print("=" * 79)


