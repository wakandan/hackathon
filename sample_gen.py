import random
import sys

file_name = sys.argv[1]
f = open(file_name, 'w')
for i in range(1000):
    rand = random.random
    tmp_str = "%s,%s,%s\n" % (str(rand()*255), str(rand()*255), str(rand()*255))
    f.write(tmp_str)

f.close()
