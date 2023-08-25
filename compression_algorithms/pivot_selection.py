import numpy as np
from time import time

class WindowReplacement:
    def window_error_2d_hist(self, x, width, error, samp_percent, bins, pool_function = False):
        self.replacements = 0
        w,l = x.shape[0], x.shape[1]
        new_x = np.copy(x)

        n = int(np.floor(width*width*samp_percent))
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                smp = np.random.choice(x[i:i+width,j:j+width].flatten(), n)
                hist, buckets = np.histogram(smp, bins = bins)
                max_bucket = np.argmax(hist)
                # midpoint of the bucket
                pivot = (buckets[max_bucket] + buckets[max_bucket+1])/2

                for k in range(i,i+width):
                    #print('k: ', k)
                    for y in range(j,j+width):
                        #print('y: ', y)
                        if np.abs(x[k,y] - pivot)< error:
                            self.replacements += 1
                            #print(k,y)
                            new_x[k,y] = pivot
                        else:
                            continue

        return new_x
    
    def window_error_2d(self, x, width, error, first_element = True, pool_function = False):
        self.replacements = 0
        w,l = x.shape[0], x.shape[1]
        new_x = np.copy(x)
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                if first_element:
                    pivot = x[i,j]
                else:
                    pivot  = np.mean(x[i:i+width,j:j+width])

                for k in range(i,i+width):
                    #print('k: ', k)
                    for y in range(j,j+width):
                        #print('y: ', y)
                        if np.abs(x[k,y] - pivot)< error:
                            self.replacements += 1
                            #print(k,y)
                            new_x[k,y] = pivot
                        else:
                            continue

        return new_x
    
if __name__ == '__main__':
    # test on a 2d array
    x =np.random.normal(0,1,(1000,1000))
    wr = WindowReplacement()

    start = time()
    new_x = wr.window_error_2d(x, 10, 0.5, first_element = True)
    end = time()

    print('Old version: ', wr.replacements)
    print('Old time: ', round(end-start, 2))

    wr = WindowReplacement()
    start = time()
    new_x = wr.window_error_2d_hist(x, 10, 0.5, 0.1, 10, pool_function = False)
    end = time()
    print('New version: ', wr.replacements)
    print('New time: ', round(end-start, 2))