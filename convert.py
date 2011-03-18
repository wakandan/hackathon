import math
import numpy
f = [t*0.01 for t in range(1001)]
a = [3*math.sin(2*math.pi * i) + 5*math.cos(8*math.pi*i) for i in f]
b = numpy.fft.fft(a)
c = [(f[i], math.sqrt(math.pow(b[i].real,2)+math.pow(b[i].imag,2))) for i in range(1001) if i<400]
def compare(x,y):
    return cmp(x[1], y[1])
c.sort(compare)
print c[-1]
