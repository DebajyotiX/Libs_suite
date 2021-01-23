import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
# from scipy.optimize import leastsq
from PyPDF2 import PdfFileMerger
from scipy.optimize import curve_fit
import math

plt.rcParams["figure.figsize"] = [8.27, 6]


# to match the width of the front page(width of plots in pdf)

def mergePDFtoMaster(name_of_pdf, not_delete=None):
    # =========================MERGING PDFs ====================
    # check if "output.pdf" file exists alreay
    if os.path.exists('./report.pdf'):
        pass
    else:
        copyfile('Designs/design 5.0 (master).pdf', 'report.pdf')

    pdfs = ['report.pdf']
    pdfs = pdfs + [name_of_pdf]
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write("report2.pdf")
    # Writing to a different name and then delete prev report and rename the current one
    merger.close()
    if not_delete == False or not_delete == None:
        os.remove(name_of_pdf)
    os.remove("report.pdf")
    os.rename("report2.pdf", "report.pdf")


class Plotter:
    general_peaks_detected_index = np.array([])

    # Will be used to store peaks so that it is available across all functions

    def __init__(self, filename, title, size=None):
        if size == None:
            last = 22492  # set default value here
        else:
            last = size  # custom data set size
        seperator = "\t"  # using wavelength is seperated from intensity in a line

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

        # Attributes of the Object
        self.xData = data[:, 0]
        self.yData = data[:, 1]
        self.title = title
        self.no_of_points = last

    def raw_plot(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            # setting temporary/local rc parameters
            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(1, 1, 1))

            # ==============================PLOTTING============================
            ax.plot(x, y, color=(0.0, 0.0, 0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax.set_facecolor((1, 1, 1))
            ax.set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax.set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax.set_title("RAW DATA", color=(0.5, 0.5, 0.5))
            ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([-4000, 40000])  # set the range here
            ax.grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)

            # =========================SAVING PLOT ====================
            plt.savefig('temp_raw.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1))
            mergePDFtoMaster('temp_raw.pdf')
            if show_plot == None or show_plot == True:
                plt.show()

    def raw_2x1_subplot(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=1, facecolor=(1, 1, 1))
            first_divide = int(size / 2)

            # =========================PLOTTING====================
            ax[0].plot(x[0:first_divide], y[0:first_divide], color=(0.0, 0.0, 0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000, 40000])  # set the range here
            ax[0].set_facecolor((1, 1, 1))
            ax[0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1].plot(x[first_divide:-1], y[first_divide:-1], color=(0.0, 0.0, 0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[1].set_xlim([x[first_divide], x[-1]])  # set the range here
            ax[1].set_facecolor((1, 1, 1))
            ax[1].set_ylim([-4000, 40000])  # set the range here
            ax[1].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[0].set_title("RAW DATA 2x1", color=(0.5, 0.5, 0.5))
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            plt.savefig('temp_2x1_subplot.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_2x1_subplot.pdf')
            if show_plot == None or show_plot == True:
                plt.show()

    def raw_3x1_subplot(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=3, ncols=1, facecolor=(1, 1, 1), sharey=True)
            first_divide = int(size / 3)
            second_divide = int(2 * size / 3)

            # =========================PLOTTING====================
            ax[0].plot(x[0:first_divide], y[0:first_divide], color=(0.0, 0.0, 0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[0].set_xlim([x[0], x[first_divide]])  # set the range here
            ax[0].set_ylim([-4000, 40000])  # set the range here
            ax[0].set_facecolor((1, 1, 1))
            ax[0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1].plot(x[first_divide:second_divide], y[first_divide:second_divide], color=(0.0, 0.0, 0.0),
                       linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[1].set_xlim([x[first_divide], x[second_divide]])  # set the range here
            ax[1].set_facecolor((1, 1, 1))
            ax[1].set_ylim([-4000, 40000])  # set the range here
            ax[1].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[2].plot(x[second_divide:-1], y[second_divide:-1], color=(0.0, 0.0, 0.0), linewidth=0.4)

            # =========================COSMETICS of plotting ====================
            ax[2].set_xlim([x[second_divide], x[-1]])  # set the range here
            ax[2].set_facecolor((1, 1, 1))
            ax[2].set_ylim([-4000, 40000])  # set the range here
            ax[2].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[2].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[2].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[2].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax[0].set_title("RAW DATA 3x1", color=(0.5, 0.5, 0.5))

            plt.savefig('temp_3x1_subplot.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_3x1_subplot.pdf')
            if show_plot == None or show_plot == True:
                plt.show()

    def raw_CHNO_peaks(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=2, facecolor=(1, 1, 1))

            # setting the X-limits for the CHNO peaks(nm)
            a1 = 245.0
            a2 = 250.0

            b1 = 630.0
            b2 = 645.0

            c1 = 740.0
            c2 = 750.0

            d1 = 774.0
            d2 = 784.0

            # Finding index of the wavelength value nearest to a1, a2, b1, b2, c1, c2, d1, d2
            index_a1 = min(range(len(x)), key=lambda i: abs(x[i] - a1))
            index_b1 = min(range(len(x)), key=lambda i: abs(x[i] - b1))
            index_c1 = min(range(len(x)), key=lambda i: abs(x[i] - c1))
            index_d1 = min(range(len(x)), key=lambda i: abs(x[i] - d1))
            index_a2 = min(range(len(x)), key=lambda i: abs(x[i] - a2))
            index_b2 = min(range(len(x)), key=lambda i: abs(x[i] - b2))
            index_c2 = min(range(len(x)), key=lambda i: abs(x[i] - c2))
            index_d2 = min(range(len(x)), key=lambda i: abs(x[i] - d2))

            # =========================PLOTTING====================
            ax[0, 0].plot(x[index_a1:index_a2], y[index_a1:index_a2], color=(0.0, 0.0, 0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[0, 0].set_xlim([x[index_a1], x[index_a2]])  # set the range here
            ax[0, 0].set_facecolor((1, 1, 1))
            ax[0, 0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[0, 0].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[0, 0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[0, 1].plot(x[index_b1:index_b2], y[index_b1:index_b2], color=(0.0, 0.0, 0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[0, 1].set_xlim([x[index_b1], x[index_b2]])  # set the range here
            ax[0, 1].set_facecolor((1, 1, 1))
            ax[0, 1].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[0, 1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1, 0].plot(x[index_c1:index_c2], y[index_c1:index_c2], color=(0.0, 0.0, 0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[1, 0].set_xlim([x[index_c1], x[index_c2]])  # set the range here
            ax[1, 0].set_facecolor((1, 1, 1))
            ax[1, 0].set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax[1, 0].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1, 0].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[1, 0].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))

            # =========================PLOTTING====================
            ax[1, 1].plot(x[index_d1:index_d2], y[index_d1:index_d2], color=(0.0, 0.0, 0.0), linewidth=0.7)

            # =========================COSMETICS of plotting ====================
            ax[1, 1].set_xlim([x[index_d1], x[index_d2]])  # set the range here
            ax[1, 1].set_facecolor((1, 1, 1))
            ax[1, 1].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            ax[1, 1].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
            ax[1, 1].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            # ax[0, 0].set_title("CHNO Peaks", color=(0.5, 0.5, 0.5))

            plt.savefig('temp_CHNO_peaks.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('temp_CHNO_peaks.pdf')
            if show_plot == None or show_plot == True:
                plt.show()

    def spectrum_peaks(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        # ==========================finding peak============================
        peak_index, properties = find_peaks(y, height=600, distance=40, prominence=500, width=4)

        Plotter.general_peaks_detected_index = peak_index
        # Assigning the calculated peaks' index to the class variable.
        # This way, these peak values can be accessed by other functions, without re-calculating it.

        print("Peaks are : \n", x[peak_index])
        print("no of peaks are:", len(peak_index) )

        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            # setting temporary/local rc parameters
            fig, ax = plt.subplots(nrows=1, ncols=1, facecolor=(1, 1, 1))

            # ==============================PLOTTING============================
            ax.plot(x, y, color=(0.0, 0.0, 0.0), linewidth=0.4)
            plt.plot(x[peak_index], y[peak_index], ".", color='#ff6400')

            # =========================COSMETICS of plotting ====================
            ax.set_facecolor((1, 1, 1))
            ax.set_ylabel('Counts', color=(0.5, 0.5, 0.5))
            ax.set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))
            title = "Total no peaks: "+str(len(peak_index))
            ax.set_title(title, color=(0.5, 0.5, 0.5),loc='left')
            ax.set_title("Peaks", color=(0.5, 0.5, 0.5),loc='center')
            ax.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([-4000, 40000])  # set the range here
            ax.grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)

            # =========================SAVING PLOT ====================
            plt.savefig('temp_raw.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1))
            mergePDFtoMaster('temp_raw.pdf')
            if show_plot == None or show_plot == True:
                plt.show()

    def selected_peaks(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points

        manually_peaks_selected = np.array([247.90, 279.61, 393.468, 396.94, 636.10, 777.50, 280.32])    #INPUT
        # peaks-jaggery from Arun bhaiya's presentation  :P

        '''
        manual_peak_index = np.array([]) #index of manual peaks
        for i in range(len(manually_peaks_selected)):
            temp = np.abs(x-np.array([manually_peaks_selected[i]]*len(x)))
            temp   = np.argmin(temp, axis=0)
            manual_peak_index = np.append(manual_peak_index, temp)
        
        manual_peak_index = manual_peak_index.astype(int)
        '''

        x_selected = np.arange(len(manually_peaks_selected))  # [1,2,3,4...]s equence for bar plot
        y_selected = np.array([])

        peak_indices, properties = find_peaks(y, height=600, distance=40, prominence=500, width=4)
        #finding algorithm-peaks

        x_peak = np.array([]) #peak index
        y_peak = np.array([])

        for i in range(len(peak_indices)):
            x_peak = np.append(x_peak, x[peak_indices[i]]) #Algorithm-peak wavelength values
            y_peak = np.append(y_peak, y[peak_indices[i]])

        index_of_manually_peaks_selected=np.array([])
        for j in range(len(manually_peaks_selected)):
            #comparing manual peaks to algorithm ones to get index
            absolute_val_array = np.abs(x_peak - manually_peaks_selected[j])
            smallest_difference_index = absolute_val_array.argmin(axis=0)
            # index = min(range(len(x)), key=lambda i: abs(x[Plotter.general_peaks_detected_index[i]] - manually_peaks_selected[j]))
            index_of_manually_peaks_selected= np.append(index_of_manually_peaks_selected, smallest_difference_index)
            # Appending the index to a list
            y_selected=np.append(y_selected, y_peak[smallest_difference_index])

        # BAR PLOT

        plt.bar(x_selected, y_selected, width=0.8, color ='#ff6400')
        # plt.set_facecolor((1, 1, 1))
        plt.ylabel('Counts',color=(0.3, 0.3, 0.3))
        plt.xlabel('wavelength(nm)',color=(0.3, 0.3, 0.3))
        plt.yticks(color=(0.3, 0.3, 0.3))
        plt.title('Selected Peaks', color=(0.3, 0.3, 0.3))
        plt.xticks(x_selected, ("247.90", "279.61", "393.468", "396.94", "636.10", "777.50", "280.32"),color =(0.3, 0.3, 0.3))        # # plt.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
        plt.savefig('selected_peaks.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                    orientation='landscape', papertype='A4', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        mergePDFtoMaster('selected_peaks.pdf')

        if show_plot == None or show_plot == True:
           plt.show()

    def noise_region(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        manually_peaks_selected = np.array([247.90, 279.61, 393.468, 396.94, 636.10, 777.50, 280.32])    #INPUT
        SNR_manually_peaks_selected = np.array([])
        # peaks-jaggery from Arun bhaiya's presentation  :P
        noise_region_limits = np.array([220,260,390,670,875])
        noise_sample = np.array([[245, 247.5], [281, 285], [300, 310], [760, 765]])

        for j in range(4):
            # noise_region_limits[j] = min(range(len(x)), key=lambda i: abs(x[i] - noise_region_limits[j]))
            noise_sample[j][0] = min(range(len(x)), key=lambda i: abs(x[i] - noise_sample[j][0]))
            noise_sample[j][1] = min(range(len(x)), key=lambda i: abs(x[i] - noise_sample[j][1]))

        stdev_noise=np.array([])
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=2, facecolor=(1, 1, 1))

            # =========================FITTING====================
            c=0
            for i in range(2):
                for  j in range(2):
                    a=int(noise_sample[c][0])
                    b=int(noise_sample[c][1])
                    x_dense =np.arange(start=x[a], stop=x[b], step=0.0001)  # Densely taken X values for plotting fit
                    stdev_noise= np.append(stdev_noise, (np.std(y[a:b])+np.mean(y[a:b])))
                    # stdev_noise= np.append(stdev_noise, np.std(y[a:b]))
                    noise_const = np.array([stdev_noise[c]]*len(x_dense))
                    # ax[i, j].hlines(y=stdev_noise[c], xmin=x[a], xmax=x[b], linewidth=2, color='r')
                    ax[i, j].plot(x_dense, noise_const, 'r--', label='Mean + Standard Deviation')
                    ax[i, j].legend(loc='lower right',fontsize='x-small')
                    # =========================PLOTTING and printing====================
                    ax[i, j].plot(x[a:b], y[a:b], color =(0.3, 0.3, 0.3))
                    # =========================COSMETICS of plotting ====================
                    ax[i, j].set_xlim(x[a], x[b])  # set the range here
                    ax[i, j].set_facecolor((1, 1, 1))
                    if j==0:
                        ax[i, j].set_ylabel('Intensity', color=(0.5, 0.5, 0.5))

                    if i==1:
                        ax[i, j].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))


                    ax[i, j].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
                    ax[i, j].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
                    c+=1


            st = fig.suptitle("Noise Regions", fontsize="x-large",color=(0.5, 0.5, 0.5))
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

            plt.savefig('noise_regions.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('noise_regions.pdf')
            if show_plot == None or show_plot == True:
                plt.show()


        # ========================= SNR for the Peaks -BAR PLOT ==============================

        # ********* finding peak values at manual peaks ******
        x_selected = np.arange(len(manually_peaks_selected))  # [1,2,3,4...]s equence for bar plot
        y_selected = np.array([])

        peak_indices, properties = find_peaks(y, height=600, distance=40, prominence=500, width=4)
        #finding algorithm-peaks

        x_peak = np.array([]) #peak index
        y_peak = np.array([])

        for i in range(len(peak_indices)):
            x_peak = np.append(x_peak, x[peak_indices[i]]) #Algorithm-peak wavelength values
            y_peak = np.append(y_peak, y[peak_indices[i]])

        index_of_manually_peaks_selected=np.array([])
        for j in range(len(manually_peaks_selected)):
            #comparing manual peaks to algorithm ones to get index
            absolute_val_array = np.abs(x_peak - manually_peaks_selected[j])
            smallest_difference_index = absolute_val_array.argmin(axis=0)
            # index = min(range(len(x)), key=lambda i: abs(x[Plotter.general_peaks_detected_index[i]] - manually_peaks_selected[j]))
            index_of_manually_peaks_selected= np.append(index_of_manually_peaks_selected, smallest_difference_index)
            # Appending the index to a list
            y_selected=np.append(y_selected, y_peak[smallest_difference_index])

        # *********** peak values found ***************

        for i in range(len(manually_peaks_selected)):  #calculating SNR at peak values
            c=0
            while(True):
                if manually_peaks_selected[i]>noise_region_limits[c] and manually_peaks_selected[i]<=noise_region_limits[c+1]:
                    SNR_manually_peaks_selected =np.append(SNR_manually_peaks_selected,y_selected[i]/stdev_noise[c])
                    break

                c+=1

        # SNR-BAR PLOT

        plt.bar(x_selected, SNR_manually_peaks_selected, width=0.8, color ='#ff6400')
        # plt.set_facecolor((1, 1, 1))
        plt.ylabel('Counts',color=(0.3, 0.3, 0.3))
        plt.xlabel('wavelength(nm)',color=(0.3, 0.3, 0.3))
        plt.yticks(color=(0.3, 0.3, 0.3))
        plt.title('Signal to noise ratio-Selected Peaks', color=(0.3, 0.3, 0.3))
        plt.xticks(x_selected, ("247.90", "279.61", "393.468", "396.94", "636.10", "777.50", "280.32"),color =(0.3, 0.3, 0.3))        # # plt.tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
        plt.savefig('SNR-selected_peaks.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                    orientation='landscape', papertype='A4', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        mergePDFtoMaster('SNR-selected_peaks.pdf')

        if show_plot == None or show_plot == True:
           plt.show()

        #end



    def lorentzian_fit(self, show_plot=None):
        x = self.xData
        y = self.yData
        name = self.title
        size = self.no_of_points
        manual_peak = np.array([247.90, 279.61, 393.468, 396.94, 636.10, 777.50, 280.32])
        range_index=np.array([])
        # setting the X-limits for the peaks(nm)

        for i in range(4):
            range_index = np.append(range_index, (manual_peak[i]-0.3))
            range_index = np.append(range_index, (manual_peak[i]+0.3))
        i+=1
        range_index = np.append(range_index, (manual_peak[i]-3))
        range_index = np.append(range_index, (manual_peak[i]+3))
        i+=1
        range_index = np.append(range_index, (manual_peak[i]-3))
        range_index = np.append(range_index, (manual_peak[i]+3))
        i+=1
        range_index = np.append(range_index, (manual_peak[i]-0.3))
        range_index = np.append(range_index, (manual_peak[i]+0.3))

        # Finding index of the wavelength value nearest to a1, a2, b1, b2, c1, c2, d1, d2
        for j in range(14):
            range_index[j] = min(range(len(x)), key=lambda i: abs(x[i] - range_index[j]))
        # =========================FITTING FUNCTION=====================
        def lorentzian_f( x, x0, a, gam ):
            return a * gam**2 / ( gam**2 + ( x - x0 )**2)


        #table
        range_index=range_index.astype(int)
        c=0
        print("x_0","-----------","amp","-----------","FWHM")
        peak_table=[]
        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=2, facecolor=(1, 1, 1))

            # =========================FITTING====================

            for i in range(2):
                for  j in range(2):
                    a=int(range_index[c])
                    b=int(range_index[c+1])
                    popt, pcov = curve_fit(lorentzian_f, x[a:b], y[a:b])
                    x_dense =np.arange(start=x[a], stop=x[b], step=0.0001)  # Densely taken X values for plotting fit
                    ax[i, j].plot(x_dense, lorentzian_f(x_dense, *popt), 'r-', label='x0=%5.2f, a=%5.0f , gam =%5.5f' % tuple(popt))
                    ax[i, j].legend(loc='lower right',fontsize='x-small')
                    # =========================PLOTTING and printing====================
                    x_0 = tuple(popt)[0]
                    amp  = tuple(popt)[1]
                    fwhm = 2*tuple(popt)[2]
                    peak_table.append((x_0,amp,fwhm))
                    print("%5.2f nm" % x_0,"------ %5.2f" % amp,"------%5.3f nm"%fwhm)
                    ax[i, j].scatter(x[a:b], y[a:b], color =(0.3, 0.3, 0.3), alpha=0.7 )
                    # =========================COSMETICS of plotting ====================
                    ax[i, j].set_xlim(x[a], x[b])  # set the range here
                    ax[i, j].set_facecolor((1, 1, 1))
                    if j==0:
                        ax[i, j].set_ylabel('Intensity', color=(0.5, 0.5, 0.5))

                    if i==1:
                        ax[i, j].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))


                    ax[i, j].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
                    ax[i, j].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
                    c+=2


            st = fig.suptitle("Selected Peaks - fit", fontsize="x-large",color=(0.5, 0.5, 0.5))
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

            plt.savefig('peak_fit1.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('peak_fit1.pdf')
            if show_plot == None or show_plot == True:
                plt.show()


        with plt.rc_context({'axes.edgecolor': ((0.8, 0.8, 0.8))}):
            fig, ax = plt.subplots(nrows=2, ncols=2, facecolor=(1, 1, 1))
            fig.delaxes(ax[1][1])
            c=8
            for i in range(2):
                for  j in range(2):
                    if c==14:
                        break

                    a=int(range_index[c])
                    b=int(range_index[c+1])
                    popt, pcov = curve_fit(lorentzian_f, x[a:b], y[a:b])
                    x_dense =np.arange(start=x[a], stop=x[b], step=0.0001)  # Densely taken X values for plotting fit
                    ax[i, j].plot(x_dense, lorentzian_f(x_dense, *popt), 'r-', label='x0=%5.2f, a=%5.0f , gam =%5.5f' % tuple(popt))
                    ax[i, j].legend(loc='lower right',fontsize='x-small')
                    # =========================PLOTTING and printing====================
                    x_0 = tuple(popt)[0]
                    amp  = tuple(popt)[1]
                    fwhm = 2*tuple(popt)[2]
                    peak_table.append((x_0,amp,fwhm))
                    print("%5.2f nm" % x_0,"------ %5.2f" % amp,"------%5.3f nm"%fwhm)
                    ax[i, j].scatter(x[a:b], y[a:b], color =(0.3, 0.3, 0.3), alpha=0.7 )
                    # =========================COSMETICS of plotting ====================
                    ax[i, j].set_xlim(x[a], x[b])  # set the range here
                    ax[i, j].set_facecolor((1, 1, 1))
                    if j==0:
                        ax[i, j].set_ylabel('Intensity', color=(0.5, 0.5, 0.5))
                    if i==1 or (i==0 and j==1):
                        ax[i, j].set_xlabel('Wavelength(nm)', color=(0.5, 0.5, 0.5))


                    ax[i, j].grid(color=(0.8, 0.8, 0.8), ls='--', lw=0.25)
                    ax[i, j].tick_params(direction='inout', length=6, width=1.5, colors=(0.5, 0.5, 0.5))
                    c+=2
                if c==14:
                    break

            st = fig.suptitle("Selected Peaks - fit",color=(0.5, 0.5, 0.5))
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

            plt.savefig('peak_fit1.pdf', dpi=300, facecolor=(1, 1, 1), edgecolor=(1, 1, 1),
                        orientation='landscape', papertype='A4', format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
            mergePDFtoMaster('peak_fit1.pdf')
        if show_plot == None or show_plot == True:
            plt.show()