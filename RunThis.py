from Libs_Suite import Libs_data
from Libs_Suite import mergePDFtoMaster


# =========================USER TWEAKABLE==================
material = Libs_data('jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc',title="Jaggery",size =22492)
# material.raw_plot(show_plot=True)
# material.raw_2x1_subplot(show_plot=True)
material.raw_3x1_subplot(show_plot=True)
material.raw_CHNO_peaks(show_plot=True)
# =========================================================


# adding a last end page(one with the bunser burner)
mergePDFtoMaster('Designs/design 5.0 (end).pdf', not_delete = True)

