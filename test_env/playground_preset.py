import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
from scipy.optimize import leastsq
from PyPDF2 import PdfFileMerger

# -------------------DATA LOADING-------------------

filename = "testdata.asc"
last = 22492  # <--------------------------EDIT HERE
seperator = "\t"
a = ""
with open(filename, "r") as file:
    # here "data" is a empty matrix of last x 2 shape
    # It will be filled by real data later on
    data = np.array(np.arange(0, last * 2, dtype=float).reshape(last, 2))
    lines = file.read().splitlines()  # array of lines
    for i in range(last):  # i is the line no(row no in data file)
        a = ""
        line = lines[i]
        for j in range(len(line)):
            if line[j] == seperator:  # first term
                data[i, 0] = float(a)
                a = ""
            elif j == len(line) - 1:  # last term
                a = a + line[j]
                data[i, 1] = float(a)
                a = ""
            else:  # just concatenate
                a = a + line[j]
# data
x = data[:, 0]
y = data[:, 1]


# --------------------TRIMMING FUNCTION ---------------------

def trimmer(x, a, b):
    index_a = min(range(len(x)), key=lambda i: abs(x[i] - a))
    index_b = min(range(len(x)), key=lambda i: abs(x[i] - b))
    return index_a, index_b


cut1 = 220  # cutoff wavelength in nm <-----------EDIT HERE
cut2 = 874  # <-----------EDIT HERE
a, b = trimmer(x, cut1, cut2)
x = x[a:b]
y = y[a:b]
ymax = max(y)
y = y / ymax
for i in range(b - a):
    if y[i] <= 0:
        y[i] = -y[i]
        # flipping noise intentionally for profiling

# plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y, linewidth=0.4)


# ----------------------------------------------------------
# ----------------------PLAYGROUND STARTS-------------------
# ----------------------------------------------------------


#FINDING PEAKS

index_1 = min(range(len(x)), key=lambda i: abs(x[i] - 260))
peak_index1, properties = find_peaks(y[0:index_1], height=0.100, distance=40, prominence=0.01, width=4)

print(peak_index1)

index_2 = min(range(len(x)), key=lambda i: abs(x[i] - 400))
peak_index2, properties = find_peaks(y[index_1:index_2], height=0.025, distance=40, prominence=0.005, width=4)
#shifting indices
for i in range(len(peak_index2)):
    peak_index2[i]=peak_index2[i]+index_1

print(peak_index2)

index_3 = min(range(len(x)), key=lambda i: abs(x[i] - 700))
peak_index3, properties = find_peaks(y[index_2:index_3], height=0.007, distance=40, prominence=0.01, width=3)
#shifting indices
for i in range(len(peak_index3)):
    peak_index3[i]=peak_index3[i]+index_2


print(peak_index3)

index_4 = min(range(len(x)), key=lambda i: abs(x[i] - 800))
peak_index4, properties = find_peaks(y[index_3:index_4], height=0.025, distance=40, prominence=0.01, width=3)
#shifting indices
for i in range(len(peak_index4)):
    peak_index4[i]=peak_index4[i]+index_3


print(peak_index4)

index_5 = min(range(len(x)), key=lambda i: abs(x[i] - 870))
peak_index5, properties = find_peaks(y[index_4:index_5], height=0.1, distance=40, prominence=0.01, width=4)
#shifting indices
for i in range(len(peak_index5)):
    peak_index5[i]=peak_index5[i]+index_4


print(peak_index5)

#concataning indices
peak_index = np.append(peak_index1 , peak_index2)
peak_index = np.append(peak_index , peak_index3)
peak_index = np.append(peak_index , peak_index4)
peak_index = np.append(peak_index , peak_index5)


#
# peak_index, properties = find_peaks(y, height=0.05, distance=50, prominence=0.01, width=4)
#
# peak_index = np.append(peak_index,1)
# for i in range(len(peak_index)):
#     peak_index[i]=peak_index[i]+20


plt.plot(x[peak_index], y[peak_index], ".", color='#ff6400')
print("Peaks are : \n", peak_index)
plt.show()
