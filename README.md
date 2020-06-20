# Libs_suite
This is a python library which automatically generates PDF reports containing plots and analysis from LIBS data.

# Try on given TEST DATA
OPEN RunThis.py
There, you can run it once to see how the code works. 
It will produce some plots and ultimately save it in a pdf file named report.py
# Try on your OWN DATA
1. To work with your own file, add your libs data file in the same directory as RunThis.py
2. Open RunThis.py and replace the given filename(with extension)

i.e.. material = Libs_data('<put your file name>.<extension>',title="<material name here>",size = <no of points>)

eg. material = Libs_data('granite_MechelleSpect.asc',title="Granite",size =22492)

3. Run the file RunThis.py. making sure you have the following libraries
    a) Mathplotlib
    b) Numpy
    c) PyPDF2
4. Delete or rename the output pdf file named #report.pdf before re running the code



