
#from typing import Int , Float
import matplotlib.pyplot as plt
class AverageMeter(object):
    def __init__(self):
        self.epoch = 0
        self.val_epoch_list = []
        self.reset()

    def reset(self):
        self.batch_list = []
        self.val_list = []
        self.max = 0
        self.min = -10000
    def return_current_avg(self):
        #print('self.val_size_list : ',self.val_size_list)
        #print('self.batch_size_list : ',self.batch_size_list)
        #print('self.val_size_list : ', len(self.val_size_list))
        #print('self.batch_size_list : ', len(self.batch_size_list))
        #print(sum(self.batch_size_list))
        #print(sum(self.val_size_list))
        return sum(self.val_list)/sum(self.batch_list) 

    def update_val_epoch_list(self):
        self.val_epoch_list.append(self.return_current_avg())
        self.max = max(self.val_epoch_list)
        self.min = min(self.val_epoch_list)
    
    def update_fold_on_min_flag(self):
        if len(self.val_epoch_list)<1:
            return True
        if self.min == self.val_epoch_list[-1]:
            return True
        return True

    def update_fold_on_max_flag(self):
        if len(self.val_epoch_list)<1:
            return True
        if self.max == self.val_epoch_list[-1]:
            return True
        return True

    def check_min_value_in_last_elements_of_queue(self, length):
        if len(self.val_epoch_list)<length:
            return True 
        list_values_to_be_checked = self.val_epoch_list[:-length]
        if self.min in list_values_to_be_checked:
            return True
        else:
            return False

    def check_max_value_in_last_elements_of_queue(self, length):
        list_values_to_be_checked = self.val_epoch_list[:-length]
        if self.max in list_values_to_be_checked:
            return True
        else:
            return False

    def update(self, val: int or float, batch_size: int , epoch: float ):
        #print('self.epoch, epoch : ', self.epoch, epoch)
        if self.epoch !=epoch:
            self.update_val_epoch_list()
            self.reset()
            self.epoch = epoch
        elif self.epoch ==epoch:
            self.val_list.append(val)
            self.batch_list.append(batch_size)
            

    def plot(self, title = 'train or test', ylabel = 'accuracy or loss', figsize=(20, 15), dpi = 300):
        plt.figure(figsize=figsize, dpi = dpi)
        plt.plot(self.val_epoch_list)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epochs')
        plt.show()
        