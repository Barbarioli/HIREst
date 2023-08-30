import numpy as np
from time import time
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class WindowReplacement:
    def __init__(self, first_element = True):
        self.first_element = first_element
    def window_error_2d(self, x, width, error):
        w,l = x.shape[0], x.shape[1]
        if self.first_element:
            for i in range(0, w-width+1, width):
                for j in range(0, l-width+1, width):
                    window = x[i:i+width,j:j+width]
                    pivot = window[0,0]
                    window[np.abs(window - pivot) < error] = pivot
                    x[i:i+width,j:j+width] = window
        else:
            for i in range(0, w-width+1, width):
                for j in range(0, l-width+1, width):
                    window = x[i:i+width,j:j+width]
                    pivot  = window[np.random.randint(0,width), np.random.randint(0,width)]
                    window[np.abs(window - pivot) < error] = pivot
                    x[i:i+width,j:j+width] = window
        return x

class WindowReplacementHistogram:
    def __init__(self, samp_percent, bins, midpoint=True):
        self.samp_percent = samp_percent
        self.bins = bins
        self.midpoint = midpoint

    def window_error_2d(self, x, width, error):
        samp_percent = self.samp_percent
        bins = self.bins
        n = int(np.floor(width*width*samp_percent))
        w,l = x.shape[0], x.shape[1]

        if self.midpoint:
            for i in range(0, w-width+1, width):
                for j in range(0, l-width+1, width):
                    window = x[i:i+width,j:j+width]
                    smp = np.random.choice(window.flatten(), n)
                    hist, buckets = np.histogram(smp, bins = bins)
                    max_bucket = np.argmax(hist)
                    pivot = (buckets[max_bucket] + buckets[max_bucket+1])/2
                    window[np.abs(window - pivot) < error] = pivot
                    x[i:i+width,j:j+width] = window
        else:
            for i in range(0, w-width+1, width):
                for j in range(0, l-width+1, width):
                    window = x[i:i+width,j:j+width]
                    smp = np.random.choice(window.flatten(), n)
                    hist, buckets = np.histogram(smp, bins = bins)
                    max_bucket = np.argmax(hist)
                    bucket_min = buckets[max_bucket]
                    bucket_max = buckets[max_bucket+1]
                    for s in range(n):
                        if smp[s] >= bucket_min and smp[s] < bucket_max:
                                pivot = smp[s]
                                break
                    window[np.abs(window - pivot) < error] = pivot
                    x[i:i+width,j:j+width] = window
        return x
    
class WindowReplacementKDE:
    def __init__(self, samp_percent):
        self.samp_percent = samp_percent
    
    def window_error_2d(self, x, width, error):
        samp_percent = self.samp_percent
        n = int(np.floor(width*width*samp_percent))
        w, l = x.shape[0], x.shape[1]
        for i in range(0, w-width+1, width):
            for j in range(0, l-width+1, width):
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
        w, l = x.shape[0], x.shape[1]
        for i in range(0, w-width+1, width):
            for j in range(0, l-width+1, width):
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
    file = np.fromfile('data/ODV_dust4_1_1800_3600.f32', dtype=float)
    file = file.reshape(1800, 1800)
    x = file[0:1024,0:1024]

    x = (x - np.min(x))/(np.max(x) - np.min(x))
    print(np.max(x), np.min(x))
    og = x.copy()
    # x =np.random.normal(0,1,(1000,1000))
    wr = WindowReplacement(first_element=True)
    window = 16
    # error = 0.001
    error = 0.2
    print("Window size: ", window)
    

    start = time()
    new_x = wr.window_error_2d(x, window, error)
    end = time()
    

    # get max error
    print(np.max(np.abs(og - new_x)))
    print('First element time: ', round(end-start, 2))

    x = og
    og = og.copy()
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    print(np.max(x), np.min(x))

    # histogram
    wr = WindowReplacementHistogram(0.1, 10)
    start = time()
    new_x = wr.window_error_2d(x, window, error)
    end = time()
    print(np.max(np.abs(og - new_x)))
    print('Histogram time: ', round(end-start, 2))




