import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# from scipy.optimize import leastsq
from PyPDF2 import PdfFileMerger

# to match the width of the front page(width of plots in pdf)
plt.rcParams["figure.figsize"] = [8.27, 6]

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

        # ==============================PLOTTING============================
        x=self.xData
        y=self.yData
        name =self.title
        with plt.rc_context({'axes.edgecolor':((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(1, 1, 1))
            ax.set_facecolor((1, 1, 1))
            ax.plot(x,y, color=(0.3,0.3,0.3), linewidth=0.5)

            # =========================COSMETICS of plotting ====================
            ax.set_ylabel('counts', color=(0.5, 0.5, 0.5))
            ax.set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            # for xmaj in ax.xaxis.get_majorticklocs():
            #     ax.axvline(x=xmaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for xmin in ax.xaxis.get_minorticklocs():
            #     ax.axvline(x=xmin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymaj in ax.yaxis.get_majorticklocs():
            #     ax.axhline(y=ymaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymin in ax.yaxis.get_minorticklocs():
            #     ax.axhline(y=ymin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            ax.set_title("RAW DATA", color=(0.5, 0.5, 0.5))
            ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))
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
            ax[0].plot(x[0:first_divide],y[0:first_divide], color=(0.3,0.3,0.3), linewidth=0.5)

            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000,40000])  # set the range here
            ax[0].set_facecolor((1, 1 , 1))

            # =========================COSMETICS of plotting ====================
            ax[0].set_ylabel('counts', color=(0.5, 0.5, 0.5))
            ax[0].grid((0.8, 0.8, 0.8), ls = '--', lw = 0.25)

            # for xmaj in ax[0].xaxis.get_majorticklocs():
            #     ax[0].axvline(x=xmaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for xmin in ax[0].xaxis.get_minorticklocs():
            #     ax[0].axvline(x=xmin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymaj in ax[0].yaxis.get_majorticklocs():
            #     ax[0].axhline(y=ymaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymin in ax[0].yaxis.get_minorticklocs():
            #     ax[0].axhline(y=ymin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))


            ax[1].plot(x[first_divide:-1], y[first_divide:-1], color=(0.3,0.3,0.3), linewidth=0.5)
            ax[1].set_xlim([x[first_divide], x[-1]])  # set the range here
            ax[1].set_facecolor((1, 1 , 1))
            ax[1].set_ylim([-4000,40000])  # set the range here

            # =========================COSMETICS of plotting ====================
            ax[1].set_ylabel('counts', color=(0.5, 0.5, 0.5))
            ax[1].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[0].set_title("RAW DATA 2x1", color=(0.5, 0.5, 0.5))
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))
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
            ax[0].plot(x[0:first_divide],y[0:first_divide], color=(0.3,0.3,0.3), linewidth=0.5)
            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000,40000])  # set the range here
            ax[0].set_facecolor((1, 1 , 1))

            # =========================COSMETICS of plotting ====================
            ax[0].set_ylabel('counts', color=(0.5, 0.5, 0.5))
            # for xmaj in ax[0].xaxis.get_majorticklocs():
            #     ax[0].axvline(x=xmaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for xmin in ax[0].xaxis.get_minorticklocs():
            #     ax[0].axvline(x=xmin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymaj in ax[0].yaxis.get_majorticklocs():
            #     ax[0].axhline(y=ymaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymin in ax[0].yaxis.get_minorticklocs():
            #     ax[0].axhline(y=ymin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            ax[0].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)

            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))


            ax[1].plot(x[first_divide:second_divide], y[first_divide:second_divide], color=(0.3,0.3,0.3), linewidth=0.5)
            ax[1].set_xlim([x[first_divide], x[second_divide]])  # set the range here
            ax[1].set_facecolor((1, 1 , 1))
            ax[1].set_ylim([-4000,40000])  # set the range here

            # =========================COSMETICS of plotting ====================
            ax[1].set_ylabel('counts', color=(0.5, 0.5, 0.5))
            # for xmaj in ax[1].xaxis.get_majorticklocs():
            #     ax[1].axvline(x=xmaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for xmin in ax[1].xaxis.get_minorticklocs():
            #     ax[1].axvline(x=xmin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymaj in ax[1].yaxis.get_majorticklocs():
            #     ax[1].axhline(y=ymaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymin in ax[1].yaxis.get_minorticklocs():
            #     ax[1].axhline(y=ymin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            ax[1].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))
           

            ax[2].plot(x[second_divide:-1], y[second_divide:-1], color=(0.3,0.3,0.3), linewidth=0.5)
            ax[2].set_xlim([x[second_divide], x[-1]])  # set the range here
            ax[2].set_facecolor((1, 1 , 1))
            ax[2].set_ylim([-4000,40000])  # set the range here

            # =========================COSMETICS of plotting ====================
            ax[2].set_ylabel('counts', color=(0.5, 0.5, 0.5))
            ax[2].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            # for xmaj in ax[2].xaxis.get_majorticklocs():
            #     ax[2].axvline(x=xmaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for xmin in ax[2].xaxis.get_minorticklocs():
            #     ax[2].axvline(x=xmin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymaj in ax[2].yaxis.get_majorticklocs():
            #     ax[2].axhline(y=ymaj, ls='-', color=(0.9, 0.9, 0.9), linewidth=0.5)
            # for ymin in ax[2].yaxis.get_minorticklocs():
            #     ax[2].axhline(y=ymin, ls='--', color=(0.9, 0.9, 0.9), linewidth=0.5)
            ax[2].grid(color=(0.8, 0.8, 0.8), ls = '--', lw = 0.25)
            ax[2].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5),
                grid_color=(0.5, 0.5, 0.5))
            ax[0].set_title("RAW DATA 3x1", color=(0.5, 0.5, 0.5))
            plt.savefig('temp_3x1_subplot.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                orientation='landscape', papertype='A4', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_3x1_subplot.pdf')
            if show_plot == None or show_plot==True:
                plt.show() 


#====================== USER TWEAKABLE =========================
# jaggery = Libs_data('jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc',title="Jaggery",size =22492)
# jaggery.raw_plot(show_plot=True)
# jaggery.raw_2x1_subplot(show_plot=True)




