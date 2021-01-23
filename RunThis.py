from Libs_Suite import Plotter
from Libs_Suite import mergePDFtoMaster


# =========================USER TWEAKABLE==================
material = Plotter('jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc',title="Jaggery",size =22492)
# material.raw_plot(show_plot=True)
# material.raw_2x1_subplot(show_plot=True)
# material.raw_3x1_subplot(show_plot=True)
# material.raw_CHNO_peaks(show_plot=True)
# material.spectrum_peaks(show_plot=True)
# material.selected_peaks(show_plot=True)
# material.noise_region(show_plot=True)
# material.lorentzian_fit(show_plot=True)
# =========================================================


# adding a last end page(one with the bunser burner)
mergePDFtoMaster('Designs/design 5.0 (end).pdf', not_delete = True)



