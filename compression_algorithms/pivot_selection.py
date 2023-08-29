import numpy as np
from time import time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class WindowReplacement:
    def window_error_2d(self, x, width, error, first_element = True, pool_function = False):
        w,l = x.shape[0], x.shape[1]
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                window = x[i:i+width,j:j+width]
                if first_element:
                    pivot = x[i,j]
                else:
                    pivot  = np.mean(window)
                window[np.abs(window - pivot) < error] = pivot
                x[i:i+width,j:j+width] = window
        return x

class WindowReplacementHistogram:
    def __init__(self, samp_percent, bins):
        self.samp_percent = samp_percent
        self.bins = bins

    def window_error_2d(self, x, width, error):
        samp_percent = self.samp_percent
        bins = self.bins
        w,l = x.shape[0], x.shape[1]
        n = int(np.floor(width*width*samp_percent))
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                window = x[i:i+width,j:j+width]
                smp = np.random.choice(window.flatten(), n)
                hist, buckets = np.histogram(smp, bins = bins)
                max_bucket = np.argmax(hist)
                # midpoint of the bucket
                pivot = (buckets[max_bucket] + buckets[max_bucket+1])/2
                window[np.abs(window - pivot) < error] = pivot
                x[i:i+width,j:j+width] = window
        return x
    
class WindowReplacementKDE:
    def __init__(self, samp_percent):
        self.samp_percent = samp_percent
    
    def window_error_2d(self, x, width, error):
        w,l = x.shape[0], x.shape[1]
        samp_percent = self.samp_percent
        n = int(np.floor(width*width*samp_percent))
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                window = x[i:i+width,j:j+width]
                smp = np.random.choice(window.flatten(), n)
                pivot = smp[np.argmax(gaussian_kde(smp).pdf(smp))]
                window[np.abs(window - pivot) < error] = pivot
                x[i:i+width,j:j+width] = window 
        return x

class WindowReplacementAdaptivePivot:
    def __init__(self, new_pivot_number):
        self.new_pivot_number = new_pivot_number
    
    def window_error_2d(self, x, width, error):
        new_pivot_number = self.new_pivot_number
        w,l = x.shape[0], x.shape[1]
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                counter = 0
                for k in range(i,i+width):
                    for y in range(j,j+width):
                        if counter % new_pivot_number == 0:
                            pivot = x[k,y]
                        if np.abs(x[k,y] - pivot)< error:
                            x[k,y] = pivot
                        counter +=1
        return x

    
    
if __name__ == '__main__':
    # test on a 2d array
    file = np.fromfile('data/CLDHGH_1_1800_3600.f32', dtype=float)
    file = file.reshape(1800, 1800)
    x = file[0:1024,0:1024]

    x = (x - np.min(x))/(np.max(x) - np.min(x))
    og = x.copy()
    # x =np.random.normal(0,1,(1000,1000))
    wr = WindowReplacement()
    window = 8
    error = 0.001
    print("Window size: ", window)

    start = time()
    new_x = wr.window_error_2d(x, window, error, first_element = True)
    end = time()

    # get max error
    print(np.max(np.abs(og - new_x)))
    print('First element time: ', round(end-start, 2))

    # wr = WindowReplacement()
    # start = time()
    # new_x = wr.window_error_2d_hist(x, window, error, sampling_prcnt, 10)
    # end = time()
    # print('Hist: ', wr.replacements)
    # print('Hist time: ', round(end-start, 2))

    # # adaptive pivot
    # wr = WindowReplacement()
    # start = time()
    # new_x = wr.window_error_2d_adaptive_pivot(x, window, 16, error)
    # end = time()
    # print('Adaptive pivot: ', wr.replacements)
    # print('Adaptive pivot time: ', round(end-start, 2))

    # # kde
    # wr = WindowReplacement()
    # start = time()
    # new_x = wr.window_error_2d_kde(x, window, error, sampling_prcnt)
    # end = time()
    # print('KDE: ', wr.replacements)
    # print('KDE time: ', round(end-start, 2))



