"""
Vectorization Comparison for Computing Sum of Squares
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import numpy as np
import timeit
import matplotlib.pylab as plt
import pandas as pd

def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """
    # TODO: Implement this function
    #numpy.random.randn
    return np.random.rand(n)


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    # TODO: Implement this function
    # for i in range(len(samples)):
    #     samples[i] = (samples[i]**2)
    sum = 0
    for num in samples:
        sum += num**2
    return sum


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    # TODO: Implement this function
    return np.dot(samples, samples)


def time_ss(sample_list):
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an 
    element in sample_list), and then time the same random 
    sample using the for and numpy loops.

    Parameters
    ----------
    samples : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.
    """
    # TODO: Implement this function
    
    lfr = []
    lnp = []
    for num in sample_list:
        tmp = gen_random_samples(num)

        a = timeit.default_timer()
        sum_squares_for(tmp)
        b = timeit.default_timer()
        sum_squares_np(tmp)
        c = timeit.default_timer()
        lfr.append(b-a)
        lnp.append(c-b)
    


    td = {
        'n': sample_list,
        'ssfor': lfr,
        'ssnp': lnp
    }

    return td


def timess_to_df(ss_dict):
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """
    # TODO: Implement this function
    df = pd.DataFrame.from_dict(ss_dict)
    return df


def main():
    # generate 100 samples
    samples = gen_random_samples(100)
    # call the for version
    ss_for = sum_squares_for(samples)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    # make sure they are approximately the same value
    import numpy.testing as npt
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)

    """
    code for the graph (3F) below
    
    #run time_ss of values from 10 to 10M 
    #rn test will b small to easily run at first
    # 10 000 000
    samples = np.linspace(10,10000000,10, dtype=int)
    #np.arange(10, 10000000).tolist()
    #np.linspace(10,10000000,3, dtype=int)
    tmp_dict = time_ss(samples)
    df = timess_to_df(tmp_dict)
    
    
    #have x be the size of list
    #have y be the time for 
    x = df['n']
    y1 = df['ssfor']
    y2 = df['ssnp']
    plt.scatter(x,y1,c = "red",marker="^", label= "ssfor")
    plt.scatter(x,y2,c = "blue", marker= "o",label= "ssnp")
    plt.xlabel('size of data (ten million)')
    plt.ylabel('time (sec)')
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    main()
