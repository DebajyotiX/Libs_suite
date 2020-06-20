import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import leastsq
from PyPDF2 import PdfFileMerger

def mergePDFtoMaster(name_of_pdf):

        # =========================MERGING PDFs ====================
        pdfs = ['master.pdf']
        pdfs = pdfs + [name_of_pdf]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        os.remove(name_of_pdf)
        os.remove('master.pdf')
        merger.write("master.pdf")
        merger.close()

class Libs_data:
    
    def __init__(self,filename,title,size=None):
        if size==None:
            last = 22492 # total no of points for ICCD data
        else:
            last = size # custom data set size
        # filename = "jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc"
        a = ""  # used for data extraction
        with open(filename, "r") as file:
            data = np.array(np.arange(0, last * 2, dtype=float).reshape(last, 2))
            # here "data" is a empty matrix of last x 2 shape(will be filled by real data later)
            lines = file.read().splitlines()  # array of lines
            for i in range(last):  # i is the line no(row no in csv)
                a = ""
                line = lines[i]
                for j in range(len(line)):
                    if line[j] == "\t":  # first term
                        data[i, 0] = float(a)
                        a = ""
                    elif j == len(line) - 1:  # last term
                        a = a + line[j]
                        data[i, 1] = float(a)
                        a = ""
                    else:  # just concatenate
                        a = a + line[j]

        self.xData = data[:,0]
        self.yData = data[:,1]
        self.title = title
        self.no_of_points = last

    def raw_plot(self,no_graph=None):

        # ==============================PLOTTING============================
        x=self.xData
        y=self.yData
        name =self.title
        fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(0.2, 0.2, 0.2))
        ax.set_facecolor((0.2, 0.2, 0.2))
        ax.plot(x,y, color='yellow', linewidth=0.5)
        ax.set_xlim([x[0], x[-1]])  # set the range here

        # =========================COSMETICS of plotting ====================
        ax.set_ylabel('Intensity(counts)', color=(0.7, 0.7, 0.7))
        ax.set_xlabel('Wavelength(nm)', color=(0.7, 0.7, 0.7))
        for xmaj in ax.xaxis.get_majorticklocs():
            ax.axvline(x=xmaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for xmin in ax.xaxis.get_minorticklocs():
            ax.axvline(x=xmin, ls='--', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymaj in ax.yaxis.get_majorticklocs():
            ax.axhline(y=ymaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymin in ax.yaxis.get_minorticklocs():
            ax.axhline(y=ymin, ls='--')
        ax.set_title("LIBS SPECTRUM - "+name+"(RAW DATA)", color=(0.7, 0.7, 0.7))
        ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.7, 0.7, 0.7),
                       grid_color=(0.7, 0.7, 0.7))
        # plt.rcParams["figure.figsize"] = [6.4 * 2, 4.8 * 1.2]
        # plt.rc('figure', figsize=(11.69, 8.27))
        # plt.tight_layout()
        plt.savefig('temp_raw.pdf', dpi=300, facecolor=(0.2, 0.2, 0.2), edgecolor=(0.2, 0.2, 0.2),
                    orientation='landscape', papertype='A4', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        mergePDFtoMaster('temp_raw.pdf')

        if no_graph == None:
            plt.show()  

    def raw_2x1_subplot(self,no_graph=None):
        x=self.xData
        y=self.yData
        name =self.title
        size=self.no_of_points

        fig, ax = plt.subplots(nrows=2, ncols=1, facecolor=(0.2, 0.2, 0.2),sharey=True)
        first_divide =int(size/2)
        ax[0].plot(x[0:first_divide],y[0:first_divide], color='yellow', linewidth=0.5)
        ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
        ax[0].set_ylim([-4000,40000])  # set the range here
        ax[0].set_facecolor((0.2, 0.2, 0.2))

        # =========================COSMETICS of plotting ====================
        ax[0].set_ylabel('Intensity(counts)', color=(0.7, 0.7, 0.7))
        for xmaj in ax[0].xaxis.get_majorticklocs():
            ax[0].axvline(x=xmaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for xmin in ax[0].xaxis.get_minorticklocs():
            ax[0].axvline(x=xmin, ls='--', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymaj in ax[0].yaxis.get_majorticklocs():
            ax[0].axhline(y=ymaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymin in ax[0].yaxis.get_minorticklocs():
            ax[0].axhline(y=ymin, ls='--')
        ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.7, 0.7, 0.7),
                       grid_color=(0.7, 0.7, 0.7))


        ax[1].plot(x[first_divide:-1], y[first_divide:-1], color='yellow', linewidth=0.5)
        ax[1].set_xlim([x[first_divide], x[-1]])  # set the range here
        ax[1].set_facecolor((0.2, 0.2, 0.2))
        ax[1].set_ylim([-4000,40000])  # set the range here

        # =========================COSMETICS of plotting ====================
        ax[1].set_ylabel('Intensity(counts)', color=(0.7, 0.7, 0.7))
        ax[1].set_xlabel('Wavelength(nm)', color=(0.7, 0.7, 0.7))
        for xmaj in ax[1].xaxis.get_majorticklocs():
            ax[1].axvline(x=xmaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for xmin in ax[1].xaxis.get_minorticklocs():
            ax[1].axvline(x=xmin, ls='--', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymaj in ax[1].yaxis.get_majorticklocs():
            ax[1].axhline(y=ymaj, ls='-', color=(0.3, 0.3, 0.3), linewidth=0.5)
        for ymin in ax[1].yaxis.get_minorticklocs():
            ax[1].axhline(y=ymin, ls='--')
        ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.7, 0.7, 0.7),
                             grid_color=(0.7, 0.7, 0.7))
        plt.rc('figure', figsize=(11.69, 8.27))
        plt.savefig('temp_2x1_subplot.pdf', dpi=300, facecolor=(0.2, 0.2, 0.2), edgecolor=(0.2, 0.2, 0.2),
                    orientation='landscape', papertype='A4', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        mergePDFtoMaster('temp_2x1_subplot.pdf')
        if no_graph == None:
            plt.show() 

jaggery = Libs_data('jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc',title="Jaggery")
jaggery.raw_plot()
jaggery.raw_2x1_subplot()

# jaggery.raw_3x1_subplot()



