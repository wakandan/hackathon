import matplotlib
import numpy
import matplotlib.pyplot as plt

def normalize(x):
    '''x is a list of values'''
    input_data = x[:]    
    N = len(input_data)
    mean = float(sum(input_data))/N
    variance = float(sum([i**2 for i in input_data]))/N - mean**2
    return [float(i-mean)/math.sqrt(variance) for i in input_data]
        
def transform(samples):
    '''Discrete Fourier Transform of an array'''

    input_data = samples[:]
    if type(input_data) != numpy.ndarray:
        input_data = numpy.array(input_data)
    return numpy.fft.fft(input_data)
    pass

def draw(y_values, x_values=None): 
    '''draw a bar graph of y_values at y axis and x_values at x axis'''
    graph_width = 0.05
    if x_values is None: 
        x_values = range(len(y_values))
    if len(x_values) != len(y_values):
        raise Exception("Input data are not in same size")
    y_values = [i.__abs__() for i in y_values]
    #print y_values
    p1 = plt.bar(x_values, y_values, graph_width)
    plt.show()
    pass
