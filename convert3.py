import math
import numpy
import mdp
import sys
import array

def main():
    file_name =  sys.argv[1]
    fs = 15 
    input_data = read_file(file_name)
    normalized_input_data = normalize_samples(input_data)
    for i in range(3):
        channel2 = apply_ica(normalized_input_data, i)
        print channel2
        print calc_heart_rate(channel2, fs)

def read_file(file_name):
    
    '''
        Receive rgb data from a file in this format r,g,b, on each line        
    '''
    a = open(file_name).read().split('\n')
    def my_split(x):
        _float = float
        return map(_float, x.split(','))
    return map(my_split, a[:-1])

def normalize_sample(x):
    '''
        x is a 1-dimensional raw list of receive data
        Return a normalized list
    '''
    
    #calc mean value. This may be not correct
    N = len(x)
    mean = float(sum(x))/N
    x1 = [i**2 for i in x]
    mean_x1 = float(sum(x1))/N
    variance = mean_x1 - mean**2
    normalized_x = [float(i-mean)/variance for i in x]
    return normalized_x
    
def normalize_samples(x):
    '''
        Normalize all the 3 channels
        x is a 3-dimensional list of raw data. Just normalize each of them
    '''

    f_n = normalize_sample
    return map(f_n, x)


def apply_ica(x, i):
    '''
        Seperate the original rgb sources into 3 different channels
        i is between 0 to 2, which is the channel
    '''

    #x1 = numpy.transpose(x)
    x1 = numpy.array(x)
    channels = mdp.fastica(x1)
    return channels[i]


def calc_heart_rate(x, fs):
    '''
        x is a discrete function, list of values. 
        Returns a DFT of these samples.
        fs is the sampling frequency        
        N is the total number of samples
    '''

    N = len(x)
    #use discrete frequency transform for the samplings
    dft_x = numpy.fft.fft(x)
    amp_max = 0
    tar_freq = 0
    for i in range(1, N):
        tmp = dft_x[i]
        f = fs * i / float(N)
        amp = math.sqrt(tmp.real**2 + tmp.imag**2)
        if amp_max < amp:
            amp_max = amp
            tar_freq = f
    return tar_freq

if __name__ == '__main__':
    main()
