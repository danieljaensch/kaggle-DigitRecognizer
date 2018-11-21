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
import time

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
# ----------------------- train -------------------------------------
# -------------------------------------------------------------------
# ---- set parameters ---------------
param_data_directory = "MNIST_data"         # default: MNIST_data
param_output_size = 10                      # 10 - original
param_save_filename = "checkpoint.pth"      # checkpoint.pth
param_save_filename_validation_loss = "checkpoint_validation_model.pth"
param_save_directory = "./"                 # ./
param_learning_rate = 0.001                 # 0.001
param_hidden_units = 512                    # 512
param_epochs = 3                            # 5
param_print_every_steps = 20                # 20
param_gpu = True                            # True or False
# -----------------------------------

print("----- running with params -----")
print("data directory: ", param_data_directory)
print("save directory: ", param_save_directory)
print("learning rate:  ", param_learning_rate)
print("hidden units:   ", param_hidden_units)
print("epochs:         ", param_epochs)
print("gpu:            ", param_gpu)
print("-------------------------------")


# --------- create model --------
model_nn = cnn.CNNNetwork( param_data_directory, param_save_filename_validation_loss, param_output_size, param_hidden_units, param_learning_rate, param_gpu )
# load previous save model from the validation loss file
#model_nn.load_state_dictionary( param_save_filename_validation_loss )


# save time stamp
start_time = time.time()
# --------- training --------
model_nn.train( param_epochs, param_print_every_steps, param_gpu )
# ---------------------------
# print duration time
print("duration: ", cnn.get_duration_in_time( time.time() - start_time ) )

# -------------- save -------
model_nn.save_model( param_save_directory + param_save_filename, param_data_directory, param_hidden_units, param_output_size, param_epochs, param_learning_rate )
# ---------------------------


# ---- test -----------------
# save time stamp
start_time = time.time()
model_nn.check_accuracy_on_test( param_gpu )
# print duration time
print("duration: ", cnn.get_duration_in_time( time.time() - start_time ) )
# ---------------------------
