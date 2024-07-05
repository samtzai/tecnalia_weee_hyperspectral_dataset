import sys
import os
# Add the main project folder path to the sys.path list

curr_path = os.path.dirname(os.path.realpath(__file__))
curr_path = curr_path.replace('\\','/')
main_path,_ = curr_path.rsplit('/',1)
print (main_path)
sys.path.append(main_path)