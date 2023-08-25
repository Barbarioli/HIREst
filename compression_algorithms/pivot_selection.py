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

    def window_error_2d_adaptive_pivot(self, x, width, new_pivot_number, error, pool_function = False):
        self.replacements = 0
        w,l = x.shape[0], x.shape[1]
        new_x = np.copy(x)
        for i in range(0, w-width+1, int(width)):
            for j in range(0, l-width+1, int(width)):
                # if first_element:
                #     pivot = x[i,j]
                # else:
                #     pivot  = np.mean(x[i:i+width,j:j+width])

                
                counter = 0
                for k in range(i,i+width):
                    for y in range(j,j+width):
                        if counter % new_pivot_number == 0:
                            pivot = x[k,y]
                        if np.abs(x[k,y] - pivot)< error:
                            self.replacements += 1
                            #print(k,y)
                            new_x[k,y] = pivot
                        counter +=1

        return new_x
    
if __name__ == '__main__':
    # test on a 2d array
    file = np.fromfile('data/CLDLOW_1_1800_3600.f32', dtype=float)
    file = file.reshape(1800, 1800)
    x = file[0:1024,0:1024]
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    # x =np.random.normal(0,1,(1000,1000))
    wr = WindowReplacement()

    start = time()
    new_x = wr.window_error_2d(x, 10, 0.1, first_element = True)
    end = time()

    print('Old version: ', wr.replacements)
    print('Old time: ', round(end-start, 2))

    wr = WindowReplacement()
    start = time()
    new_x = wr.window_error_2d_hist(x, 10, 0.1, 0.1, 10, pool_function = False)
    end = time()
    print('New version: ', wr.replacements)
    print('New time: ', round(end-start, 2))

    # adaptive pivot
    wr = WindowReplacement()
    start = time()
    new_x = wr.window_error_2d_adaptive_pivot(x, 10, 8, 0.1, pool_function = False)
    end = time()
    print('Adaptive pivot: ', wr.replacements)
    print('Adaptive pivot time: ', round(end-start, 2))



