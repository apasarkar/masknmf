import functools
import multiprocessing
import os


def runpar(f, X, nprocesses=None, **kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    
    #Change affinity (if needed) to enable full multicore processing
    
    
    val = len(os.sched_getaffinity(os.getpid()))
    print("the CPU affinity BEFORE runpar is {}".format(val))

    if nprocesses is None:
        nprocesses = int(multiprocessing.cpu_count()) 
        print("the number of processes is {}".format(nprocesses))
#         val = len(os.sched_getaffinity(os.getpid()))
#         print('the number of usable cpu cores is {}'.format(val))
    
    with multiprocessing.Pool(initializer=parinit, processes=nprocesses) as pool:
        res = pool.map(functools.partial(f, **kwargs), X)
    pool.join()
    pool.close()


    val = len(os.sched_getaffinity(os.getpid()))
    print("after the multicore, the affinity is {}".format(val))

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))
    val = len(os.sched_getaffinity(os.getpid()))
    print("the cpu affinity after the process (intro fix) is {}".format(val))
    return res

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    num_cpu = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (num_cpu, os.getpid()))
