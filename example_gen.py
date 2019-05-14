from utils import *

with open('examples/test.ori') as f, open('examples/test','w') as fout:
    for idx, x in enumerate(f):
        line = x.split(',')
        mhc = mhc_rename(line[0])
        fout.write(','.join([mhc, line[1]])+'\n')
