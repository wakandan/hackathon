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
    xs = [i for i in xs if i> 40 and i < 240]
    x1 = int(xs[0]*N/(fs*60))
    x2 = int(xs[-1]*N/(fs*60))    

    #plt.plot(xs, x[0], color='red')
    #plt.plot(xs, x[1], color='green')    
    #plt.plot(xs, x[2], color='blue')
    print len(xs)
    print len(x[1][x1:x2+1])
    plt.plot(xs, x[0][x1:x2+1], color='red')
    plt.plot(xs, x[1][x1:x2+1], color='green')    
    plt.plot(xs, x[2][x1:x2+1], color='blue')
    plt.show()
