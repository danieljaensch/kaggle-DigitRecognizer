# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:28:49 2018

@author: Daniel Jaensch
# link: https://docs.python.org/3/tutorial/classes.html
"""
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import utilities as utils
from PIL import Image

# -------------------------------------------------------------------
# -------------- global constants -----------------------------------
# -------------------------------------------------------------------

CONST_OUTPUT_TRAIN_FOLDER = "./MNIST_data/train/"
CONST_OUTPUT_TEST_FOLDER = "./MNIST_data/test/"
CONST_OUTPUT_VALID_FOLDER = "./MNIST_data/valid/"

CONST_DATA_NN_FOLDER = "./data/"


# -------------------------------------------------------------------
# -------------- main program ---------------------------------------
# -------------------------------------------------------------------
def clear_screen():
    """
    clear the screen of the console.
    """
    if os.name in ('nt','dos'):
         os.system("cls")
    elif os.name in ('linux','osx','posix'):
         os.system("clear")
    else:
       print("\n") * 120
 


def save_vector_image(vecImage, filename):
    """
    Save the image vector in the given filename as grayscale image with 28x28 pixels.
    
    @param vecImage:      numpy array of the image
    @param filename:      the filename of the image
    """
    imconvert = np.array( vecImage )
    imconvert.resize( 28, 28 )
    im = Image.fromarray( imconvert )
    im = im.convert('L')    # convert to gray scale
    im.save( filename )


# ------------- clear screen -------------------------
# now, to clear the screen
clear_screen()

# --- create train folders ----
if utils.exist_directory( CONST_OUTPUT_TRAIN_FOLDER ) == False:
    utils.create_directory( CONST_OUTPUT_TRAIN_FOLDER )

for i in range( 10 ):
    fn = CONST_OUTPUT_TRAIN_FOLDER + "{}".format( i ) 
    if utils.exist_directory( fn ) == False:
        utils.create_directory( fn )
# ----------------------------


# --- create test folders ----
if utils.exist_directory( CONST_OUTPUT_TEST_FOLDER ) == False:
    utils.create_directory( CONST_OUTPUT_TEST_FOLDER )
    
for i in range( 10 ):
    fn = CONST_OUTPUT_TEST_FOLDER + "{}".format( i ) 
    if utils.exist_directory( fn ) == False:
        utils.create_directory( fn )
# ----------------------------
        
       
# --- create valid folders ----
if utils.exist_directory( CONST_OUTPUT_VALID_FOLDER ) == False:
    utils.create_directory( CONST_OUTPUT_VALID_FOLDER )
    
for i in range( 10 ):
    fn = CONST_OUTPUT_VALID_FOLDER + "{}".format( i ) 
    if utils.exist_directory( fn ) == False:
        utils.create_directory( fn )
# ----------------------------
        
        
# ------------- Loading The Dataset -------------------------
# loading the dataset.......(Train)
print("---------------------------------------------------------------------")
print("loading train-data: train.csv ... ", end="")
train_data = pd.read_csv( CONST_DATA_NN_FOLDER + "train.csv" )
print("done.")
print("train_data (lines x columns) : ", train_data.shape)
print("---------------------------------------------------------------------")
train_data_length = train_data.shape[0]

## loading the dataset.......(Test)
##print("---------------------------------------------------------------------")
##print("loading test-data: test.csv ...", end="")
##test_data = pd.read_csv( CONST_DATA_NN_FOLDER + "test.csv")
##print("done.")
##print("test_data (lines x columns) : ", test_data.shape)
##print("---------------------------------------------------------------------")
##test_data_length = test_data.shape[0]

# prepare loaded data
x_train_data = ( train_data.iloc[:,1:].values ).astype('float32')  # all pixel values
y_train_data = train_data.iloc[:,0].values.astype('int32')         # only labels i.e targets digits
##x_test_data = test_data.values.astype('float32')


# --------------------------------------------------------------------
# ---------- train all data from the beginning to the end ------------
count = 0
for i in range( len(x_train_data) ):
    # ---- save image ----
    if count < 4:
        fn = CONST_OUTPUT_TRAIN_FOLDER + "{}/train_image_{}.jpg".format( y_train_data[i],i )
    elif count == 4:
        fn = CONST_OUTPUT_TEST_FOLDER + "{}/test_image_{}.jpg".format( y_train_data[i],i )
    elif count > 4:
        fn = CONST_OUTPUT_VALID_FOLDER + "{}/valid_image_{}.jpg".format( y_train_data[i],i )
        count = 0
        
    count += 1
    save_vector_image( x_train_data[i], fn )
    # --------------------
    
    # print out the percent
    percent = int( (i/(train_data_length-1))*100 )
    print("\rpercent: {} %\t------ file: {}\t".format(percent, fn), end='')
    # ---------------------

print("\n----------------- training complete --------------------")
# --------------------------------------------------------------------


## --------------------------------------------------------------------
## ---------- test all data from the beginning to the end -------------
#for i in range( len(x_test_data) ):
##    # ---- save image ----
##    fn = CONST_OUTPUT_TEST_FOLDER + "0/test_image_{}.jpg".format( i )
##    save_vector_image( x_test_data[i], fn )
##    percent = int( (i/(test_data_length-1))*100 )
##    print("\rpercent: {} %\t------ file: {}".format(percent, fn), end='')
##    # ---------------------

##print("\n----------------- training complete --------------------")
## --------------------------------------------------------------------
