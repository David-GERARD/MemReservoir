import multiprocessing as mp
import numpy as np

def f(x):
    return x*x

if __name__ == '__main__':

    # Create a pool of workers
    pool = mp.Pool(mp.cpu_count())

    # Generate a list of numbers from 0 to 9
    numbers = np.arange(10)

    

    # Apply the function f to each number in the list
    results = pool.map(f, numbers)
    pool.close()
    pool.join()

    print(results)

    