# -*- coding: utf-8 -*-
"""
Spyder Editor
----------------------------------------------------
file:   main_predict.py

author: Daniel Jaensch
email:  daniel.jaensch@gmail.com
data:   2018-11-18
----------------------------------------------------
"""
# ------ imports -----
import os

import cnnnetwork as cnn
# --------------------


# -------------------------------------------------------------------
# -------------- helper functions -----------------------------------
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


# -------------------------------------------------------------------
# -------------- main program ---------------------------------------
# -------------------------------------------------------------------

# ------------- clear screen -------------------------
# now, to clear the screen
clear_screen()


# -------------------------------------------------------------------
# ----------------------- predict -----------------------------------
# -------------------------------------------------------------------
# set data_directory from the argument line
# ---- set parameters ---------------
param_data_directory = "MNIST_data"                           # default: MNIST_data
param_output_size = 10                                        # 10 - original
param_save_filename_validation_loss = "checkpoint_validation_model.pth"
param_save_directory = "./"                                   # ./
param_learning_rate = 0.001                                   # 0.001
param_hidden_units = 512                                      # 512
param_gpu = True                                              # True or False

# ---- set parameters ---------------
param_image_file = "./MNIST_data/test/6/test_image_1914.jpg"  # default: ./MNIST_data/test/6/test_image_1914.jpg 
param_load_file_name = "checkpoint.pth"                       # default: checkpoint.pth
param_top_k = 5                                               # 5


print("----- running with params -----")
print("data directory: ", param_data_directory)
print("save directory: ", param_save_directory)
print("learning rate:  ", param_learning_rate)
print("hidden units:   ", param_hidden_units)
print("gpu:            ", param_gpu)
print("-------------------------------")
print("----- running with params -----")
print("image file:     ", param_image_file)
print("load file:      ", param_load_file_name)
print("top k:          ", param_top_k)
print("-------------------------------")


# --------- create model --------
model_nn = cnn.CNNNetwork( param_data_directory, param_save_filename_validation_loss, param_output_size, param_hidden_units, param_learning_rate, param_gpu )
model_nn.load_state_dictionary( param_save_filename_validation_loss )


# ------------------ prediction ----
print("--- prediction ---")
top_probabilities, top_labels = model_nn.predict( param_image_file, param_top_k )

for i in range( len(top_labels) ):
    # add +1 to index, because the index i starts with 0
    print(" {} with {:.3f} is {}".format( i+1, top_probabilities[i], top_labels[i] ))
print("------------------")
# ----------------------------------
cnn.imshow( cnn.process_image( param_image_file ))

