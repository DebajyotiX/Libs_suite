import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
# from scipy.optimize import leastsq
from PyPDF2 import PdfFileMerger

plt.rcParams["figure.figsize"] = [8.27, 6]
# to match the width of the front page(width of plots in pdf)

def mergePDFtoMaster(name_of_pdf,not_delete = None):

        # =========================MERGING PDFs ====================
        #check if "output.pdf" file exists alreay
        if os.path.exists('./report.pdf'):
            pass
        else:
            copyfile('Designs/design 5.0 (master).pdf', 'report.pdf')

        pdfs = ['report.pdf']
        pdfs = pdfs + [name_of_pdf]
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        # os.remove('report.pdf')
        merger.write("report2.pdf")###write to a different name and then delete prev report and rename_the current one
        merger.close()
        if not_delete == False or not_delete == None:
            os.remove(name_of_pdf)
        os.remove("report.pdf")
        os.rename("report2.pdf","report.pdf")

class Libs_data:
    def __init__(self,filename,title,size=None):
        if size==None:
            last = 22492 # set default value here
        else:
            last = size # custom data set size
        seperator = "\t" # using wavelength is seperated from intensity in a line

        a = ""  # used for data extraction
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

        #Attributes of the Object
        self.xData = data[:,0]
        self.yData = data[:,1]
        self.title = title
        self.no_of_points = last

    def raw_plot(self,show_plot=None):
        x=self.xData
        y=self.yData
        name =self.title
        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            #setting temporary/local rc parameters
            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(1, 1, 1))

            # ==============================PLOTTING============================
            ax.plot(x,y, color=(0.0,0.0,0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax.set_facecolor((1, 1, 1))
            ax.set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax.set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax.set_title("RAW DATA", color=(0.5, 0.5, 0.5))
            ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([-4000,40000])  # set the range here
            ax.grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)

            # =========================SAVING PLOT ====================
            plt.savefig('temp_raw.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1))
            mergePDFtoMaster('temp_raw.pdf')
            if show_plot == None or show_plot==True:
                plt.show()  

    def raw_2x1_subplot(self,show_plot=None):
        x=self.xData
        y=self.yData
        name =self.title
        size=self.no_of_points
        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=1, facecolor=(1, 1, 1))
            first_divide =int(size/2)

            # =========================PLOTTING====================
            ax[0].plot(x[0:first_divide],y[0:first_divide], color=(0.0,0.0,0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000,40000])  # set the range here
            ax[0].set_facecolor((1, 1 , 1))
            ax[0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1].plot(x[first_divide:-1], y[first_divide:-1], color=(0.0,0.0,0.0), linewidth=0.4)
        
            # =========================COSMETICS of plotting ====================
            ax[1].set_xlim([x[first_divide], x[-1]])  # set the range here
            ax[1].set_facecolor((1, 1 , 1))
            ax[1].set_ylim([-4000,40000])  # set the range here            
            ax[1].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0].set_title("RAW DATA 2x1", color=(0.5, 0.5, 0.5))
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            
            plt.savefig('temp_2x1_subplot.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                orientation='landscape', papertype='A4', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_2x1_subplot.pdf')
            if show_plot == None or show_plot==True:
                plt.show()

    def raw_3x1_subplot(self,show_plot=None):
        x=self.xData
        y=self.yData
        name =self.title
        size=self.no_of_points
        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=3, ncols=1, facecolor=(1, 1, 1),sharey=True)
            first_divide =int(size/3)
            second_divide = int(2*size/3)

            # =========================PLOTTING====================
            ax[0].plot(x[0:first_divide],y[0:first_divide], color=(0.0,0.0,0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000,40000])  # set the range here
            ax[0].set_facecolor((1, 1 , 1))
            ax[0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))


            # =========================PLOTTING====================
            ax[1].plot(x[first_divide:second_divide], y[first_divide:second_divide], color=(0.0,0.0,0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[1].set_xlim([x[first_divide], x[second_divide]])  # set the range here
            ax[1].set_facecolor((1, 1 , 1))
            ax[1].set_ylim([-4000,40000])  # set the range here
            ax[1].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
           
            # =========================PLOTTING====================
            ax[2].plot(x[second_divide:-1], y[second_divide:-1], color=(0.0,0.0,0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[2].set_xlim([x[second_divide], x[-1]])  # set the range here
            ax[2].set_facecolor((1, 1 , 1))
            ax[2].set_ylim([-4000,40000])  # set the range here
            ax[2].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[2].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[2].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[2].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax[0].set_title("RAW DATA 3x1", color=(0.5, 0.5, 0.5))

            plt.savefig('temp_3x1_subplot.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                orientation='landscape', papertype='A4', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_3x1_subplot.pdf')
            if show_plot == None or show_plot==True:
                plt.show() 

    def raw_CHNO_peaks(self,show_plot=None):
        x=self.xData
        y=self.yData
        name =self.title
        size=self.no_of_points
        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=2, facecolor=(1, 1, 1))
            
            #setting the X-limits for the CHNO peaks(nm)
            a1 = 245.0
            a2 = 250.0

            b1 = 630.0
            b2 = 645.0

            c1 = 740.0
            c2 = 750.0
            
            d1 = 774.0
            d2 = 784.0

            # findin index of the wavelength value nearest to a,b,c,d
            index_a1=min(range(len(x)), key = lambda i: abs(x[i]-a1))
            index_b1=min(range(len(x)), key = lambda i: abs(x[i]-b1))
            index_c1=min(range(len(x)), key = lambda i: abs(x[i]-c1))
            index_d1=min(range(len(x)), key = lambda i: abs(x[i]-d1))
            index_a2=min(range(len(x)), key = lambda i: abs(x[i]-a2))
            index_b2=min(range(len(x)), key = lambda i: abs(x[i]-b2))
            index_c2=min(range(len(x)), key = lambda i: abs(x[i]-c2))
            index_d2=min(range(len(x)), key = lambda i: abs(x[i]-d2))

            # =========================PLOTTING====================
            ax[0,0].plot(x[index_a1:index_a2],y[index_a1:index_a2], color=(0.0,0.0,0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[0,0].set_xlim([x[index_a1], x[index_a2]])  # set the range here
            ax[0,0].set_facecolor((1, 1 , 1))
            ax[0,0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0,0].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0,0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))


            # =========================PLOTTING====================
            ax[0,1].plot(x[index_b1:index_b2],y[index_b1:index_b2], color=(0.0,0.0,0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[0,1].set_xlim([x[index_b1], x[index_b2]])  # set the range here
            ax[0,1].set_facecolor((1, 1 , 1))
            ax[0,1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0,1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
           
            # =========================PLOTTING====================
            ax[1,0].plot(x[index_c1:index_c2],y[index_c1:index_c2], color=(0.0,0.0,0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[1,0].set_xlim([x[index_c1], x[index_c2]])  # set the range here
            ax[1,0].set_facecolor((1, 1 , 1))
            ax[1,0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1,0].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1,0].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[1,0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1, 1].plot(x[index_d1:index_d2],y[index_d1:index_d2], color=(0.0,0.0,0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[1, 1].set_xlim([x[index_d1], x[index_d2]])  # set the range here
            ax[1, 1].set_facecolor((1, 1 , 1))
            ax[1, 1].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1, 1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[1, 1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            # ax[0, 0].set_title("CHNO Peaks", color=(0.5, 0.5, 0.5))

            plt.savefig('temp_CHNO_peaks.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                orientation='landscape', papertype='A4', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_CHNO_peaks.pdf')
            if show_plot == None or show_plot==True:
                plt.show() 

    def spectrum_peaks(self,show_plot=None):
        x=self.xData
        y=self.yData
        name =self.title
        size=self.no_of_points
        #==========================finding peak============================
        peaks, properties = find_peaks(y,height=600,distance=30,prominence=4, width=4)
        print(peaks)

        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            #setting temporary/local rc parameters
            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(1, 1, 1))

            # ==============================PLOTTING============================
            ax.plot(x,y, color=(0.0,0.0,0.0), linewidth=0.4)
            plt.plot(x[peaks], y[peaks], ".", color='#ff6400')

            # =========================COSMETICS of plotting ====================
            ax.set_facecolor((1, 1, 1))
            ax.set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax.set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax.set_title("RAW DATA", color=(0.5, 0.5, 0.5))
            ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([-4000,40000])  # set the range here
            ax.grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)

            # =========================SAVING PLOT ====================
            plt.savefig('temp_raw.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1))
            mergePDFtoMaster('temp_raw.pdf')
            if show_plot == None or show_plot==True:
                plt.show()
