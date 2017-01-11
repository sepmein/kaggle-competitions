import csv as csv
import numpy as np
csv_file_object = csv.reader(open('./train.csv', 'rb'))
header = csv_file_object.next()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)
newData =
