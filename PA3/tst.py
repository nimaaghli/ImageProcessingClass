from joblib import Parallel, delayed
from PIL import Image
import numpy as np
import cmath
from pylab import *

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
def calc(m,n,kk,ll):
    s = complex(0)
    s+=img_float32[m,n]*exp(-2j*np.pi*((kk*m/M)+(ll*n/N)))
    return s
if __name__ == '__main__':
    image = Image.open("png.png").convert('L')
    img_float32 =np.asarray(image, dtype=float)

    M = np.size(image, 0)#generate height and 
    N = np.size(image, 1)#width to loop over them 
    out = np.zeros((M, N), dtype=complex)
            
    Parallel(n_jobs=4, verbose=5)(delayed(calc)(m=x, n=y,kk=k,ll=l) for k in range(0, M) for l in range(0, N) for x 
in range(0, M) for y in range(0, N)) 

    dft_shift = np.fft.fftshift(out)
    magnitude_spectrum = 20*np.log(np.abs(dft_shift))        
    figure()
    plt.imshow(magnitude_spectrum, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
    title('Original Image')
    show()
