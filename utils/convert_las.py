import sys
import traceback
import laspy
import os
import argparse

parser = argparse.ArgumentParser(description='An auxiliary script to convert .laz files to .las (required by GEDICorrect).')

parser.add_argument('--las_dir', required=True, help='Local directory of .laz files to convert.', type=str)

args = parser.parse_args()

in_dir = args.las_dir

try:    
    def convert_laz_to_las(in_laz, out_las):
        las = laspy.read(in_laz)
        las = laspy.convert(las)
        las.write(out_las)        
    
    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for inFile in filenames:
            if inFile.endswith('.laz'):	
                in_laz = os.path.join(dirpath,inFile)
                
                out_las = in_laz.replace('laz', 'las') 
                print('working on file: ',out_las)
                convert_laz_to_las(in_laz, out_las)
                             
    print('Finished without errors - LAZ to LAS')
except:
    tb = sys.exc_info()[2]
    tbinfo = traceback.format_tb(tb)[0]
    print('Error in read_xmp.py')
    print ("PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1]))    