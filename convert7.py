import math
import numpy
import mdp
import sys
import array
import matplotlib.pyplot as plt
from pprint import pprint

min_bps = 0.75
max_bps = 4

def main():
    global min_bps
    global max_bps
    file_name =  sys.argv[1]
    fs = float(sys.argv[2])
    if float(fs)/2<max_bps:  max_bps = float(fs)/2
    input_data = read_file(file_name)
    for i in range(len(input_data)-3):
        normalized_input_data = normalize_samples(input_data[i:i+3])
        #g_modes = ['pow3', 'tanh', 'gaus', 'skew']
        g = 'gaus'
        xs = []
        ys = []
        def my_compare(x,y):
            return cmp(y[1], x[1])
        #after_ica = mdp.fastica(numpy.array(normalized_input_data))
        tmp_input_data = numpy.array(normalized_input_data)
        jade_node = mdp.nodes.JADENode(white_parm={'svd':True})
        try:
            jade_node.train(tmp_input_data)
            after_ica = jade_node.get_projmatrix()
        except mdp.NodeException, e:
            print e
            continue 
        print '-----------------'
        pprint(normalized_input_data)
        pprint(after_ica)
        pprint(jade_node.get_recmatrix())
        a = numpy.matrix(after_ica)
        b = numpy.matrix(jade_node.get_recmatrix())
        pprint(a*b)
        
        before_fft = numpy.transpose(after_ica) 
        after_fft = numpy.fft.fft(before_fft)
        pprint(after_fft)
        #x = after_fft 
        #ys = []
        #for i in x:
        #    tmp = [math.sqrt(j.real**2+j.imag**2) for j in i]
        #    ys.append(tmp)
        #xs = range(len(x[0]))
        #xs = [fs*i*60/len(x[0]) for i in xs]
        #print ys 
        #plt.plot(xs, ys[0], color='red')
        #plt.plot(xs, ys[1], color='green')    
        #plt.plot(xs, ys[2], color='blue')
        #plt.show()


    
    #for j in range(3):
        #channel2 = apply_ica(normalized_input_data, j, g)
        #print channel2
        #x = calc_heart_rate(channel2, fs)
        #xs.append([i[0]*60 for i in x])
        #ys.append([i[1] for i in x])
        #if j == 0:
        #    plt.plot(xs[j], ys[j], color='red')
        #elif j == 1:
        #    plt.plot(xs[j], ys[j], color='green')
        #else:
        #    plt.plot(xs[j], ys[j], color='blue')

        #global for_drawing
        #N = len(for_drawing)
        #for_drawing = [i for i in for_drawing if i[0]>min_bps and i[0]<max_bps]

        #values = [i[1] for i in for_drawing]
        #plt.plot([i[0]*60 for i in for_drawing], values)
    #plt.show()

def read_file(file_name):
    
    '''
        Receive rgb data from a file in this format r,g,b, on each line        
    '''
    a = open(file_name).read().split('\n')
    def my_split(x):
        _float = float
        return map(_float, x.split(','))
    tmp = map(my_split, a[:-1])
    return tmp

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


def apply_ica(x, i, g):
    '''
        Seperate the original rgb sources into 3 different channels
        i is between 0 to 2, which is the channel
        g is the mode to converge
    '''

    #x1 = numpy.transpose(x)
    x1 = numpy.array(x)
    #channels = mdp.fastica(x1, input_dim=3)
    #channels = mdp.fastica(x1, input_dim=3, verbose=True, g=g)
    channels = mdp.fastica(x1, input_dim=3, g=g)
    tmp = channels.transpose()
    return tmp[i]


def calc_heart_rate(x, fs):
    '''
        x is a discrete function, list of values. 
        Returns a DFT of these samples.
        fs is the sampling frequency        
        N is the total number of samples
    '''
    global min_bps
    global max_bps
    N = len(x)
    #use discrete frequency transform for the samplings
    dft_x = numpy.fft.fft(x)
    amp_max = 0
    tar_freq = 0

    tmp_list = [(fs*i/float(N),math.sqrt(dft_x[i].real**2 + dft_x[i].imag**2)) for i in range(N) if (fs*i/float(N))>min_bps and (fs*i/float(N))<max_bps]
    return tmp_list
    #global for_drawing
    #for_drawing = tmp_list[:]
    #for i in range(1, N):
    #    tmp = dft_x[i]
    #    f = fs * i / float(N)
    #    if f<min_bps or f>max_bps: continue
    #    amp = math.sqrt(tmp.real**2 + tmp.imag**2)
    #    if amp_max < amp:
    #        amp_max = amp
    #        tar_freq = f
    #return tar_freq

if __name__ == '__main__':
    main()
