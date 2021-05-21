from keras.callbacks import Callback
from keras.models import load_model, save_model
import os

# custom Callback function
# after every epoch saves the model with unique name
class CheckPointer(Callback):
    
    # net_name: name of the network, it will be used for save directory and file name too
    # path: base path for the output
    # nb: length of number indicating version (because: '10' is before '2', so we save as '02')
    def __init__(self, net_name, path = '..\\weights\\', nb = 2):        
        self.net_name = net_name.upper() # using uppercases, so net_name is not case sensitive
        self.path = path + self.net_name + '\\' # create path to the folder where models will be saved
        self.nb = nb
        # if the output folder doesn't exist create it
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    # this function is called after every epoch
    def on_epoch_end(self, batch, logs={}):
        # list of previos saved networks
        previous_nets = [x for x in os.listdir(self.path) if self.net_name in x]
        if len(previous_nets) == 0: # if there were no previous, this is the first
            last_number = 0 # so set last number to 0
        else: # if there were some previous networks
            # get last number from the last network name from the previous_nets list
            last_number = int(previous_nets[-1][len(self.net_name) + 1:len(self.net_name) + 1 + self.nb])
        new_number = str(last_number + 1) # new number is the last number plus one, and converted to string
        while len(new_number) < self.nb: # while our number doesn't have enough digits
            new_number = '0' + new_number # append a zero to the left
        # our output model name will be: 'NETNAME_XX.hdf5'
        model_name = self.net_name + '_' + new_number + '.hdf5'
        # save model to the appropriate folder with the appropriate name
        save_model(self.model, self.path + model_name)
        # print some feedback
        print('Model saved:', self.path + model_name)

# loads the last model with the given name
# saved models should follow the naming method of 'CheckPointer'
# net_name: name of the net
def load_last_model(net_name, path = '..\\weights\\'):
    net_name = net_name.upper() # using uppercases, so net_name is not case sensitive
    # if the output folder doesn't exist create it
    if not os.path.exists((path + net_name + '\\')):
        os.makedirs((path + net_name + '\\'))
    # list of previous saved networks
    previous_nets = [x for x in os.listdir(path + net_name + '\\') if net_name in x]
    if len(previous_nets) == 0: # if there were no previous
        # print some feedback
        print('ERROR: There is no previous models with this name: ', net_name)
        return None
    else: # if we have at least one previous model
        # print some feedback
        print('Loading model: ', previous_nets[-1])
        # return with the loaded model
        # path: 'weights/NETNAME/NETNAME_XX.hdf5'
        return load_model(path + net_name + '\\' + previous_nets[-1])