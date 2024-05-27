import numpy as np

def signalSloper(x, slope):
    # Ensure x is a 2D array
    x = np.atleast_2d(x)

    # Check if input is a column vector
    qTrans = x.shape[1] == 1
    if qTrans:
        x = x.T

    # Get dimensions
    r, N = x.shape
    numUniquePts = (N + 1) // 2

    # Create index array
    idx = np.arange(1, numUniquePts + 1)

    # Perform FFT along the second dimension
    X = np.fft.fft(x, axis=1)
    #X = X[:, :numUniquePts] * np.concatenate([np.zeros((r, 1)), (idx[:-1] ** (-slope / 2)).reshape(1, -1)], axis=1)
    X = X[:, :numUniquePts] * np.concatenate([ np.zeros((r, 1)),  np.tile(idx[:-1]**(-slope/2),(r,1))], axis = 1)
    # Handle even and odd N
    if N % 2 == 0:  # Even N includes Nyquist point
        X = np.concatenate([X, np.conj(X[:, -2:0:-1])], axis=1)
    else:  # Odd N excludes Nyquist point
        X = np.concatenate([X, np.conj(X[:, -1:0:-1])], axis=1)

    # Perform inverse FFT
    y = np.real(np.fft.ifft(X, axis=1))

    # Subtract mean and divide by standard deviation along the second dimension
    y -= np.mean(y, axis=1, keepdims=True)
    y /= np.std(y, axis=1, keepdims=True)

    # Transpose if originally a column vector
    if qTrans:
        y = y.T

    return y