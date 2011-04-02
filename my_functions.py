import numpy
import math
import mdp
import matplotlib.pyplot as plt

def normalize(x):
    '''x is a multi-dimensional array of m rows and n cols'''
    y = x.transpose()
    result = []
    N = len(y[0])
    for row in y:         
        mean = float(sum(row))/N
        variance = float(sum([i**2 for i in row]))/N - mean**2
        result.append([float(i-mean)/math.sqrt(variance) for i in row])
        
    return numpy.array(result).transpose() 

def calc_heart_rate(input_data):
    '''Input data is a list of 3-item list'''
    
    data = normalize(numpy.array(input_data))
    #try:
    #    data_ica = mdp.fastica(data, white_parm={'svd': True})
    #except mdp.NodeException, e:
    #    return None 
    #data_fft_before = data_ica.transpose()
    data_fft_before = data.transpose()
    data_fft = numpy.fft.fft(data_fft_before)
    return data_fft

def extract_frequency(complex_list):
    global fs
    amplitudes = [i.__abs__() for i in complex_list]
    return amplitudes
    #max = 0.0
    #index = -1
    #for i in range(1, len(amplitudes) + 1):
        #if max < amplitudes[i - 1]:
            #max = amplitudes[i - 1]
            #index = i
    #return fs * index / len(amplitudes)

def plot_diagrams(data, fs):
    '''data will be a fft_ed data from calc heart rate'''
   
    x = map(extract_frequency, data)
    xs = range(len(data[0]))
    N = len(xs)
    xs = [i*fs*60/float(N) for i in xs]
    i1 = int(math.ceil(40*float(N)/(fs*60)))
    i2 = int(math.floor(240*float(N)/(fs*60)))
    y = x[1][i1:i2+1]
    xs = xs[i1:i2+1]
    heart_rate_index = y.index(max(y))
    heart_rate = xs[heart_rate_index]
    print y
    print xs
    print heart_rate 
    
