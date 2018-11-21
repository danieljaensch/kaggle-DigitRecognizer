# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:04:23 2018

@author: bdjaens1
"""

import os
from pathlib import Path
 

# checks if the file with file_name exists at the absolute folder
def exist_file( folder, file_name ):
    fn = Path( folder + "\\" + file_name )
    result = False
    if fn.is_file():
        result = True
    return result



# checks if the file with file_name exists at the absolute folder
def exist_file_in_current_directory( file_name ):
    fn = Path( get_current_path() + "\\" + file_name )
    result = False
    if fn.is_file():
        result = True
    return result

    
# get the current working directory path
def get_current_path():
    return os.path.realpath('.')


    
# checks if the folder exist
# folder = absoute path toe the folder
def exist_directory( folder ):
    return os.path.isdir( folder )



# create directory as the current working directory
# param folder = 'new_sub_folder'
def create_directory( folder ):
    try:
        os.makedirs( folder )
    except:
        pass


# folder is the absolute path to the folder
def change_directory( folder ):
    os.chdir( folder )


