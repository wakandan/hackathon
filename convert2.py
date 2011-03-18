import math
import numpy
alen = 1001
f = [t*0.01 for t in range(alen)]
a = numpy.random.random(alen)
b = numpy.fft.fft(a)
c = [(f[i], math.sqrt(math.pow(b[i].real,2)+math.pow(b[i].imag,2))) for i in range(alen) if i<400 and i!=0]
def compare(x,y):
    return cmp(x[1], y[1])
c.sort(compare)
print c[-1]
