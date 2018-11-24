# -*- coding: utf-8 -*-
"""
Spyder Editor
----------------------------------------------------
file:   cnnnetwork.py

author: Daniel Jaensch
email:  daniel.jaensch@gmail.com
data:   2018-11-18
----------------------------------------------------
"""
# ------ imports -----
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import datetime

import torch
from torch import nn

from torch import optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn.functional as F
# --------------------


class Net(nn.Module):
    # choose an architecture, and complete the class
    def __init__(self, hidden_units, param_output_size):
        """
        Choose and initial the CNN architecture.
    
        @param hidden_units:       number of hidden units in the fully connected layers
        @param param_output_size:  number of output
        """
        super(Net, self).__init__()
        print("create model ... ", end="")
        # convolutional layer (sees 28x28x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # convolutional layer (sees 28x28x16 tensor)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        # convolutional layer (sees 28x28x32 tensor)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # convolutional layer (sees 28x28x64 tensor)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # linear layer (sees 14x14x24 -> 1024)
        self.fc1 = nn.Linear(in_features=14 * 14 * 24, out_features=hidden_units)
        # linear layer (1024 -> 10)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=param_output_size)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        # batch norm 
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_units)
        # output
        self.output = nn.LogSoftmax(dim=1)
        print("done")
    
    
    def forward(self, x):
        """
        Forward the vector x to the neural network and calculate the output vector x.
        
        @return:    output vector of the neural network
        """
        # add sequence of convolutional and max pooling layers
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        #x = self.pool( x )
        x = F.relu( self.conv3(x) )
        x = F.relu( self.conv4(x) )
        x = self.pool( x )
        
        # flatten image input --> 14 * 14 * 24 = 4704
        x = x.view(x.size(0), -1)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function and batch norm
        x = F.relu( self.batch_norm( self.fc1(x)) )
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu( self.fc2(x) )
        # output
        x = self.output(x)
        
        return x


# -------------------------------------------------------------------
# -------------- class Neural Network -------------------------------
# -------------------------------------------------------------------
class CNNNetwork(nn.Module):
# ------------ init --------------------------------------
    def __init__(self, param_data_directory, param_filename_save_validation_loss, param_output_size, hidden_units, learning_rate, is_gpu):
        """
        Initialize the model and all train / test / valid dataloaders
    
        @param param_data_directory:                data dictionary for the data loaders
        @param param_filename_save_validation_loss: filename for the validation-loss-save-file
        @param param_output_size:                   number of output
        @param hidden_units:                        number of hidden units in the fully connected layers
        @param learning_rate:                       the learning rate
        @param is_gpu:                              is gpu available and usable or not
        """
        super(CNNNetwork, self).__init__()
        print("cnn neural network ... ")
        print("load image data ... ", end="")
        
        # define transforms for the training data and testing data
        self.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomRotation(20),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        
        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        
        # pass transforms in here, then run the next cell to see how the transforms look
        self.train_data = datasets.ImageFolder( param_data_directory + '/train', transform=self.train_transforms )
        self.test_data  = datasets.ImageFolder( param_data_directory + '/test', transform=self.test_transforms )
        self.valid_data = datasets.ImageFolder( param_data_directory + '/valid', transform=self.test_transforms )

        self.trainloader = torch.utils.data.DataLoader( self.train_data, batch_size=32, shuffle=True )
        self.testloader  = torch.utils.data.DataLoader( self.test_data, batch_size=16 )
        self.validloader = torch.utils.data.DataLoader( self.valid_data, batch_size=16 )
        print("done")
        
        # create model
        self.model = Net(hidden_units, param_output_size)
        
        # train a model with a pre-trained network
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam( self.model.parameters(), lr=learning_rate )
        
        self.learning_rate = learning_rate
        self.filename_save_validation_loss = param_filename_save_validation_loss
        
        if is_gpu and torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self. model)
            
        print("initialized.")



    def train(self, epochs, print_every, is_gpu):
        """
        Train the model based on the train-files
    
        @param epochs:       number of epochs to train the neural network
        @param print_every:  number of iteration amount, when should printed out the actual results
        @param is_gpu:       is gpu available and usable or not
        """
        if is_gpu:
            print("start training in -gpu- mode ... ")
        else:
            print("start training in -cpu- mode ... ")

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        print_every = print_every
        steps = 0
    
        # change to cuda in case it is activated
        if is_gpu:
            self.model.cuda()
    
        self.model.train() # ---------- put model in training mode -------------------
        
        for e in range(0, epochs):
            running_loss = 0
            # call the learning rate adjustment function
            self.adjust_learning_rate(e+1, 2)

            for ii, (images, labels) in enumerate( self.trainloader ):
                steps += 1
    
                if is_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                images, labels = Variable(images), Variable(labels)
    
                self.optimizer.zero_grad()
    
                # -- forward and backward passes --
                outputs = self.model( images )
                loss = self.criterion( outputs, labels )
                loss.backward()
                self.optimizer.step()
                # ---------------------------------
    
                running_loss += loss.item()
    
                # ----- output ----
                if steps % print_every == 0:
                    # make sure network is in eval mode for inference
                    self.model.eval() # ------------- put model in evaluation mode ----------------
                    
                    # turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.validation( is_gpu )
                    
                    training_loss = running_loss / print_every
                    valid_loss = test_loss / len(self.validloader)
                    valid_accuracy = accuracy / len(self.validloader)
                    
                    print("epoch: {}/{}.. ".format( e+1, epochs ),
                          "training loss: {:.4f}.. ".format( training_loss ),
                          "validation loss: {:.4f}.. ".format( valid_loss ),
                          "validation accuracy: {:.4f}\t".format( valid_accuracy ), end="")
                    
                    running_loss = 0
                    
                    # -----------------------------
                    # save the model if validation loss has decreased
                    if valid_loss <= valid_loss_min:
                        self.save_state_dictionary( self.filename_save_validation_loss )
                        valid_loss_min = valid_loss
                    else:
                        print("")
                    # -----------------------------
                    
                    # make sure training is back on
                    self.model.train() # ---------- put model in training mode -------------------
                # -----------------
            # ------- for trainloader ----------            
        # ------- for epochs ----------
        print("-- done --")
    
    

    def validation(self, is_gpu):
        """
        Calculate the validation based on the valid-files and return the test-loss and the accuracy
    
        @param is_gpu:       is gpu available and usable or not
        """
        test_loss = 0
        accuracy = 0
        
        if is_gpu:
            self.model.cuda()
            
        for ii, (images, labels) in enumerate( self.validloader ):
            if is_gpu:
                images, labels = images.cuda(), labels.cuda()
                    
            images, labels = Variable(images), Variable(labels)

            output = self.model( images )
            test_loss += self.criterion( output, labels ).item()
    
            ps = torch.exp( output )
            equality = ( labels.data == ps.max(dim=1)[1] ) # give the highest probability
            accuracy += equality.type( torch.FloatTensor ).mean()
        # ------- for validloader ----------
        return test_loss, accuracy
        
    
    
    def check_accuracy_on_test(self, is_gpu):
        """
        Calculate the accuracy based on the test-files and print it out in percent
    
        @param is_gpu:       is gpu available and usable or not
        """
        print("calculate accuracy on test ... ", end="")
        correct = 0
        total = 0
        
        if is_gpu:
            self.model.cuda()
        
        self.model.eval() # ------------- put model in evaluation mode ----------------
        
        with torch.no_grad():
            for ii, (images, labels) in enumerate( self.testloader ):
                
                if is_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                images, labels = Variable(images), Variable(labels)
                        
                outputs = self.model( images )
                _, predicted = torch.max( outputs.data, 1 )
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # ------- for testloader ----------
        print("done.")
        print('accuracy of the network on the 10000 test images: {:.3f} %'.format(100.0 * correct / total))
    


    def adjust_learning_rate(self, epoch, lr_decay_epoch=3):
        """
        Create a learning rate adjustment function that divides the learning rate by 10 every <xy> epochs
    
        @param epochs:          number of epochs to train the neural network
        @param lr_decay_epoch:  threshold to increse the learning rate every <x> times
        """
        lr = self.learning_rate
        
        if epoch > 14*lr_decay_epoch:
            lr = lr / 10000000000
        elif epoch > 12*lr_decay_epoch:
            lr = lr / 1000000000
        elif epoch > 10*lr_decay_epoch:
            lr = lr / 100000000
        elif epoch > 8*lr_decay_epoch:
            lr = lr / 10000000
        elif epoch > 6*lr_decay_epoch:
            lr = lr / 1000000
        elif epoch > 5*lr_decay_epoch:
            lr = lr / 100000
        elif epoch > 4*lr_decay_epoch:
            lr = lr / 10000
        elif epoch > 3*lr_decay_epoch:
            lr = lr / 1000
        elif epoch > 2*lr_decay_epoch:
            lr = lr / 100
        elif epoch > lr_decay_epoch:
            lr = lr / 10
            
        # set the new learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
                
        print("adjust learning rate in epoch {} to {}".format(epoch, lr))
    
    
    
    def predict(self, image_file, topk ):
        """
        Calculate the topk prediction of the given image-file and 
        return the probabilities and labels
    
        @param image_file:      image file name and path to it
        @param topk:            the number of first top matches
        @return:                the probabilities and labels
        """
        # ------ load image data -----------
        img_np = process_image(image_file)
        # ----------------------------------
        print("get prediction ... ", end="")
        
        # prepare image tensor for prediction
        img_tensor = torch.from_numpy( img_np ).type(torch.FloatTensor)
        # add batch of size 1 to image
        img_tensor.unsqueeze_(0)
        
        # probs
        self.model.eval() # ------------- put model in evaluation mode ----------------
        
        with torch.no_grad():
            image_variable = Variable( img_tensor )
            outputs = self.model( image_variable )
        
        # top probs
        top_probs, top_labs = outputs.topk( topk )
        top_probs = torch.exp( top_probs )
        top_probs = top_probs.detach().numpy().tolist()[0] 
        top_labs = top_labs.detach().numpy().tolist()[0]
        
        print("done.")
        return top_probs, top_labs
        
    
    # ----------------
    def load_state_dictionary(self, filename):
        """
        Helper function to load state_dict only 
        
        @param filename:      the filename of load file
        """
        print("load model state dict ...", end="")
        self.model.load_state_dict(torch.load( filename ))
        print(" done.")
    
    
    def save_state_dictionary(self, filename):
        """
        Helper function to save state_dict only
        
        @param filename:      the filename of load file
        """
        print("saving model ...")
        torch.save(self.model.state_dict(), filename)
    # ----------------
    
    
    def save_model(self, filename, data_directory, hidden_units, output_size, epochs, learning_rate ):
        """
        Save the trained model in a file
        
        @param filename:             the filename of load file
        @param data_directory:       data dictionary for the data loaders
        @param hidden_units:         number of hidden units in the fully connected layers
        @param output_size:          number of output
        @param epochs:               number of epochs to train the neural network
        @param learning_rate:        the learning rate
        """
        print("save model to: ", filename, end="")
        checkpoint = {'hidden_units': hidden_units,
                      'learning_rate': learning_rate,
                      'output_size': output_size,
                      'data_directory': data_directory,
                      'epochs': epochs,
                      'optimizer_state_dict': self.optimizer.state_dict,
                      'state_dict': self.model.state_dict()}
        torch.save(checkpoint, filename)
        print(" ... done")



# -------------------------------------------------------------------
# -------------- helper functions -----------------------------------
# -------------------------------------------------------------------
def get_duration_in_time( duration ):
    """
    Calculate the duration in hh::mm::ss and return it
    
    @param duration:      timestamp from the system
    @return:              formatted string with readable hours, minutes and seconds
    """
    seconds = int( duration % 60 )
    minutes = int( (duration / 60) % 60 )
    hours   = int( (duration / 3600) % 24 )
    output = "{:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds)
    return output
    


def get_current_date_time():
    """ Return the current date and time
    
    @return:              current date and time as string
    """
    utc_dt = datetime.datetime.now(datetime.timezone.utc) # UTC time
    dt = utc_dt.astimezone() # local time
    return str(dt)



def load_model( filename, is_gpu ):
    """
    Load the trained model from the file and create a model from this and return it
    
    @param filename:    the filename
    @param is_gpu:      is gpu available and usable or not
    """
    print("load model from: ", filename)
    checkpoint = torch.load(filename)
    model = CNNNetwork(checkpoint['data_directory'], checkpoint['output_size'], checkpoint['hidden_units'], checkpoint['learning_rate'], is_gpu)
    model.load_state_dict(checkpoint['state_dict'])
    print(" ... done")
    
    return model


def plot_bargraph( np_probs, np_object_names ):
    """
    Plot an bar graph wich displayed the recognized probabilities and objects
    
    @param np_probs:            list of recognized probabilities
    @param np_object_names:     list of recognized objects
    """
    y_pos = np.arange( len(np_object_names) )
    
    plt.barh(y_pos, np_probs, align='center', alpha=0.5)
    plt.yticks(y_pos, np_object_names)
    plt.gca().invert_yaxis()        # invert y-axis to show the highest prob at the top position
    plt.xlabel("probability from 0 to 1.0")
    plt.title("Objects")
    plt.show()


def process_image( image_filename ):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    
    @param image_filename:      filename of a loadable image
    @return:                    normalized loaded image as numpy array
    """
    # process a PIL image for use in a PyTorch model
    print("load image data ... ", end="")
    # define transforms for the training data and testing data
    prediction_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    img_pil = Image.open( image_filename )
    #img_pil = img_pil.convert('L')     # convert to grayscale, 1 channel
    img_pil = img_pil.convert('RGB')    # convert to RGB, 3 channel
    img_tensor = prediction_transforms( img_pil )
    print("done")
    return img_tensor.numpy()


def imshow(image, ax=None, title=None):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    
    @param image:   the loaded image
    @param ax:      ax of plt.subplot
    @param title:   title string of this image
    @return:        subplot of plt
    """    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# -------------------------------------------------------------------
# -------------------------------------------------------------------