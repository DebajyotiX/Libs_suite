from Libs_Suite import Libs_data
from Libs_Suite import mergePDFtoMaster


# =========================USER TWEAKABLE==================
# change name of file and title, give the total no of points in size.
# make sure the data is seperated by \t i.e. tabs seperated values
# else cahange the value of the variable seperator ="," or  in Libs_suite.py
# eg seperator =","
# eg seperator =" "  # space seperated
material = Libs_data('jaggery_ss_50g_20us_70mj_1_MechelleSpect.asc',title="Jaggery",size =22492)
material.raw_plot(show_plot=True)
material.raw_2x1_subplot(show_plot=True)
# =========================================================

# adding a last end page(one with the bunser burner)
mergePDFtoMaster('Designs/design 4.0 (end).pdf', not_delete = True)

