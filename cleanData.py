import numpy as np
import csv as csv

def data(file):
    test_file = csv.reader(open(file,'rb'))
    testdata = []
    for row in test_file:
        testdata.append(row)
    testdatanp = np.array(testdata)
    y = testdatanp.astype(np.float)
    return y

