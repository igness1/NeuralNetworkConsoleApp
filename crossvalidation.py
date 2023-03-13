import numpy as np
import expections

class cross_validation_5_fold:

    def __init__(self, input_data, output_data, set_number=1):
        self.input_data = input_data  # Input data.
        self.output_data = output_data  # Output data.
        self.set_number = set_number  # Set number.

        # Data split point.
        self.index1 = int(len(self.input_data) * 0.2)
        self.index2 = int(len(self.input_data) * 0.4)
        self.index3 = int(len(self.input_data) * 0.6)
        self.index4 = int(len(self.input_data) * 0.8)

        # 5 sets of input data.
        self.i_bufor1 = input_data[:self.index1]
        self.i_bufor2 = input_data[self.index1:self.index2]
        self.i_bufor3 = input_data[self.index2:self.index3]
        self.i_bufor4 = input_data[self.index3:self.index4]
        self.i_bufor5 = input_data[self.index4:]

        # 5 sets of output data.
        self.o_bufor1 = output_data[:self.index1]
        self.o_bufor2 = output_data[self.index1:self.index2]
        self.o_bufor3 = output_data[self.index2:self.index3]
        self.o_bufor4 = output_data[self.index3:self.index4]
        self.o_bufor5 = output_data[self.index4:]

    def get_train_set(self):# Combining sets to make complete training sets.
            if self.set_number == 1:
                input = np.concatenate((self.i_bufor1, self.i_bufor2, self.i_bufor3, self.i_bufor4), axis=0)
                output = np.concatenate((self.o_bufor1, self.o_bufor2, self.o_bufor3, self.o_bufor4),
                                        axis=0)
            elif self.set_number == 2:
                input = np.concatenate((self.i_bufor2, self.i_bufor3, self.i_bufor4, self.i_bufor5), axis=0)
                output = np.concatenate((self.o_bufor2, self.o_bufor3, self.o_bufor4, self.o_bufor5),
                                        axis=0)
            elif self.set_number == 3:
                input = np.concatenate((self.i_bufor3, self.i_bufor4, self.i_bufor5, self.i_bufor1), axis=0)
                output = np.concatenate((self.o_bufor3, self.o_bufor4, self.o_bufor5, self.o_bufor1),
                                        axis=0)
            elif self.set_number == 4:
                input = np.concatenate((self.i_bufor1, self.i_bufor2, self.i_bufor4, self.i_bufor5), axis=0)
                output = np.concatenate((self.o_bufor1,self.o_bufor2, self.o_bufor4, self.o_bufor5),
                                        axis=0)
            elif self.set_number == 5:
                input = np.concatenate((self.i_bufor1, self.i_bufor2, self.i_bufor3, self.i_bufor5), axis=0)
                output = np.concatenate((self.o_bufor1, self.o_bufor2, self.o_bufor3, self.o_bufor5),
                                        axis=0)
            else:
                raise Expections.WrongSetNumberChoosen
            return input, output

    def get_test_set(self): # Selection of training set.
            if self.set_number == 1:
                input = self.i_bufor5
                output = self.o_bufor5
            elif self.set_number == 2:
                input = self.i_bufor1
                output = self.o_bufor1
            elif self.set_number == 3:
                input = self.i_bufor2
                output = self.o_bufor2
            elif self.set_number == 4:
                input = self.i_bufor3
                output = self.o_bufor3
            elif self.set_number == 5:
                input = self.i_bufor4
                output = self.o_bufor4
            else:
                raise Expections.WrongSetNumberChoosen
            return input, output

