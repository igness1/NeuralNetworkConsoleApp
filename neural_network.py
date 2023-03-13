import numpy as np
import math
import matplotlib.pyplot as plt
import closest_value
import expections
from activation_functions import ActivationFunctions
from normalization import MinMaxScaler, MinMaxScalerList

class neuralNetwork:

    def __init__(self, input_train_data, output_train_data, input_test_data, output_test_data, target_classes, number_of_neurons_in_hidden_layers=[10, 10, 10],learning_rate=0.001, biasAdded=True, activation_function="sigmoid"):
        # set neural network parameters
        self.number_of_neurons_in_hidden_layers = number_of_neurons_in_hidden_layers #defined number of neurons in hidden layers
        self.number_of_hidden_layers = len(self.number_of_neurons_in_hidden_layers) #defined number of hidden layers
        self.number_of_outputs = 1 #number of outputs, not to be defined by user for now
        self.learning_rate = learning_rate #learning rate
        self.momentum = 0.9 #momentum hiperparameter 
        self.biasAdded = biasAdded #if bias is added
        self.target_classes = target_classes # values of classification classes

        """
          Assigned input and output values
          for the trainig and testing set.
        """
        self.input_layer_train, self.output_layer_train = input_train_data, output_train_data
        self.input_layer_test, self.output_layer_test = input_test_data, output_test_data

        """
          Setting the selected activation function by using ActivationFunction class instance,
          that contains function definitions.
        """
        activation_functions = ActivationFunctions()
        self.activation_function = activation_functions.function_dict[activation_function]
        self.activation_function_derivative = activation_functions.deriviate_dict[activation_function]

        self.number_of_inputs = len(self.input_layer_train[1]) # number of inputs

        self.all_neurons = [] # all neurons in network
        self.weights = [] # initial weights
        self.init_weights = [] # initial weights after updates
        
        # values ​​updating weights in earlier iterations needed for momentum
        self.weight_increment_previous_iteration = [None] * (self.number_of_hidden_layers + 1)
        # all errors in network, for all neurons in layers
        self.all_errors_in_network = [None] * (self.number_of_hidden_layers + 1)
        self.errors_ME = [] # Mean Error during training
        self.error_RMSE = [] # Root Mean during training
        self.error_MSE = [] # Mean squared Error during training
        self.bias = 1 # bias
        
        # bias impulses
        bias_column_train = np.array([self.bias] * len(self.input_layer_train))
        bias_column_test = np.array([self.bias] * len(self.input_layer_test))
        self.bias_column_train, self.bias_column_test = bias_column_train.reshape(-1, 1), bias_column_test.reshape(-1, 1)
        # Unique output layer values.
        # File 'closest_value' contains 'unique' fuctions that is imported.
        self.unique_list_of_targets_class = closest_value.unique(self.output_layer_train)
        self.accuracy_validation_list = [] # Network accuracy list during learning.
        self.fail_accuracy = False # if network accuracy is failing, condition to stop learning process
        self.result_accuracy_on_test_data = 0 # Final accuracy on test data

    def error_in_output_layer(self, output, expected_output, value_for_deriv):
        return  (expected_output - output) * self.activation_function_derivative(value_for_deriv)

    def error_in_hidden(self,previous_output_layer, weights, value_for_deriv):
        return  (previous_output_layer @ weights.T) * self.activation_function_derivative(value_for_deriv)

    def update_weights(self, inputs, errors_in_layer):
        weight_difference = (inputs @ errors_in_layer) * self.learning_rate
        return weight_difference

    def update_weights_with_momentum(self, inputs, errors_in_layer, weight_increment_previous_iteration):
        weight_difference = (inputs @ errors_in_layer) * self.learning_rate
        weight_difference = weight_difference + (self.momentum * weight_increment_previous_iteration)
        return weight_difference

    def initial_weights(self):
        if self.biasAdded: # if bias is added, then the weights matrix will have an extra row.
            self.number_of_inputs += self.bias
            # weights between the input layer and the hidden layers
            for i in range(self.number_of_hidden_layers):
                self.weights.append(
                    2 * np.random.random((self.number_of_inputs, self.number_of_neurons_in_hidden_layers[i])) - 1)
                self.number_of_inputs = self.number_of_neurons_in_hidden_layers[i] + self.bias
            # weights between last hidden layer and output layer
            self.weights.append(
                2 * np.random.random(
                    ((self.number_of_neurons_in_hidden_layers[-1] + self.bias), self.number_of_outputs)) - 1)
        else: # if bias is not added
            # weights between the input layer and the hidden layers
            for i in range(self.number_of_hidden_layers):
                self.weights.append(
                    2 * np.random.random((self.number_of_inputs, self.number_of_neurons_in_hidden_layers[i])) - 1)
                self.number_of_inputs = self.number_of_neurons_in_hidden_layers[i]
            # weights between last hidden layer and output layer
            self.weights.append(
                2 * np.random.random((self.number_of_neurons_in_hidden_layers[-1], self.number_of_outputs)) - 1)
        # assinging initial weights values to variable init_weights,
        # to always have access to original version of weights.
        self.init_weights = self.weights

    def accuracy(self, input_layer, output_layer, validation_through_learning=False):
        # output layer result for every pattern 
        all_outputs = []
        # bias column
        bias_column = np.array([self.bias] * len(input_layer))
        # for each sample/pattern in the input data neurons value is calculated.
        for pattern in range(len(input_layer)):
            neurons_in_network_ = []
            input = np.append(input_layer[pattern], bias_column[pattern]) if self.biasAdded else \
                input_layer[pattern]
            input = np.reshape(input, (-1, 1))
            # calculating neurons in each layers
            for j in range(self.number_of_hidden_layers + 1):
                # sum of input impulses and weights passed as an argument for activation function
                activation_func = self.activation_function((input.T @ self.init_weights[j]))
                neurons_in_network_.append(activation_func)
                input = np.append(neurons_in_network_[j], bias_column[pattern]) if self.biasAdded else \
                    neurons_in_network_[j]

                input = np.reshape(input, (-1, 1))
            
            """
                Function 'closest' from file 'closest_value' signes actual output value to nearest class value.
                Then compare the actual value with expected one.
            """
            closest_val = closest_value.closest(self.unique_list_of_targets_class, neurons_in_network_[-1])
            # output layers neurons are saved as a result
            all_outputs.append(closest_val)

        points = 0 # points number achieved by network
        MaxPoints = len(all_outputs) # max points number
        for i in range(MaxPoints):
            """
            If the accuracy is not validate during training process, then values are displayed on the screen.
            """
            if validation_through_learning == False:
                
                max = len(self.target_classes)-1
                # the class names for the actual values
                actual_value_class = MinMaxScalerList(feature_range=(0,max)).fit_transform(all_outputs)
                # the class names for the expected values
                expected_value_class = MinMaxScaler(feature_range=(0,max)).fit_transform(output_layer)
                
                sample_nb = f"Sample: {i}"
                print(sample_nb.center(30,"-"))
                print("  Actual value: ", all_outputs[i] ," --> ", self.target_classes[int(actual_value_class[i])])
                print("  Expected value: ", output_layer[i]," --> ", self.target_classes[int(expected_value_class[i])])
            # If actual == expected --> +1 point.
            if all_outputs[i] == output_layer[i]:
                points += 1
  
        return points / MaxPoints * 100

    def train_network(self, epochs = 1000):
        # number of epochs/iterations
        for epoch in range(epochs+1):
            # Sum of difference value between actual and expected value for samples in given epoch.
            sum_of_difference = []
            # The samples are given one by one.
            # Weights update is processed after every sample.
            for pattern in range(len(self.input_layer_train)):
                # Neurons in layers in sample.
                neurons_in_layers = []
                # 
                # value_for_derivative -> sum of weights and inputs/impulses, that is after used in activation function.
                value_for_derivative = []
                # All errors in network in sample/pattern.
                all_errors_in_net = [None] * (self.number_of_hidden_layers + 1)
                # Add bias input if biasAdded == True.
                if self.biasAdded:
                    input = np.append(self.input_layer_train[pattern], self.bias_column_train[pattern])
                    input = np.reshape(input, (-1, 1))
                else:
                    input = self.input_layer_train[pattern]
                    input = np.reshape(input, (-1, 1))
                # Process of calculating neurons in every layer.
                for j in range(self.number_of_hidden_layers + 1):
                    # Sum of inputs + weights.
                    synapses_sum = input.T @ self.init_weights[j]
                    # Activation function calculation.
                    activation_func = self.activation_function(synapses_sum)
                    # Add neurons in layer to list.
                    neurons_in_layers.append(activation_func)
                    # Add sum of weights + inputs to list for activation function derivative.
                    value_for_derivative.append(synapses_sum)
                    # Add bias inputs as an additional neuron to actual layer, because actual layer is an input layer for next layer.
                    input = np.append(neurons_in_layers[j], self.bias_column_train[pattern]) \
                        if self.biasAdded else \
                        neurons_in_layers[j]
                    input = np.reshape(input, (-1, 1))

                # Permanent addition of bias to layers for learning process.
                if self.biasAdded:
                    for j in range(self.number_of_hidden_layers):
                        neurons_in_layers[j] = np.append(neurons_in_layers[j], self.bias_column_train[pattern])
                        neurons_in_layers[j] = np.reshape(neurons_in_layers[j], (1, -1))
                        value_for_derivative[j] = np.append(value_for_derivative[j],self.bias_column_train[pattern])
                        value_for_derivative[j] = np.reshape(value_for_derivative[j],(1, -1))

                # Actual output value.
                output = neurons_in_layers[self.number_of_hidden_layers]
                # Error in output layer.
                error_output_layer = self.error_in_output_layer(output, self.output_layer_train[pattern],value_for_derivative[-1])
                error_output_layer = np.reshape(error_output_layer, (-1, 1))
                # Add difference between actual and expected to list for ME, RMSE, MSE.
                sum_of_difference.append(self.output_layer_train[pattern] - output)
                # Output layer error is added to list for errors in nerwork for this sample/patter. Needed for weights update.
                all_errors_in_net[self.number_of_hidden_layers] = error_output_layer
                # Actual layer error is the previous layer error for the next layer.
                prev_error_layer = all_errors_in_net[self.number_of_hidden_layers]

                # Calculating the errors in hidden layers. Backpropagation.
                for k in range(1, self.number_of_hidden_layers + 1):
                    num = (self.number_of_hidden_layers + 1) - k
                    all_errors_in_net[num - 1] = self.error_in_hidden(prev_error_layer, self.init_weights[num], value_for_derivative[num - 1])
                    if self.biasAdded:
                        all_errors_in_net[num - 1] = np.delete(all_errors_in_net[num - 1], -1, axis=1)
                    prev_error_layer = all_errors_in_net[num - 1]

                if self.biasAdded:
                    inputs = np.append(self.input_layer_train[pattern], self.bias_column_train[pattern])
                    inputs = np.reshape(inputs, (-1, 1))
                else:
                    inputs = self.input_layer_train[pattern]
                    inputs = np.reshape(inputs, (-1, 1))

                # Weights update.
                for w in range(self.number_of_hidden_layers + 1):
                    if epoch > 0:
                        weight_incrmnt = self.update_weights_with_momentum(inputs, all_errors_in_net[w], self.weight_increment_previous_iteration[w])
                        self.init_weights[w] += weight_incrmnt
                        self.weight_increment_previous_iteration[w] = weight_incrmnt
                    else:
                        weight_incrmnt = self.update_weights(inputs, all_errors_in_net[w])
                        self.init_weights[w] += weight_incrmnt
                        self.weight_increment_previous_iteration[w] = weight_incrmnt

                    inputs = neurons_in_layers[w].T


            sum_of_difference = np.reshape(sum_of_difference, (-1, 1))
            # Calculating ME error.
            ME = np.mean(abs(sum_of_difference))
            # Calculating MSE error.
            MSE = np.mean((sum_of_difference)**2)
            # Calculating RMSE error.
            RMSE = math.sqrt(MSE)
            #Lists to observe how errors changed during learning.
            self.errors_ME.append(ME)
            self.error_RMSE.append(RMSE)
            self.error_MSE.append(MSE)

            # Accuracy checking by using 'self.accuracy'..
            accuracy = self.accuracy(self.input_layer_test, self.output_layer_test, True)
            self.accuracy_validation_list.append(accuracy)

            # STOP learning condition.
            try:
                if epoch > 5:
                    fails = 0
                    for i in range(5):
                        if self.accuracy_validation_list[epoch - i] < self.accuracy_validation_list[epoch - (i + 1)]:
                            fails += 1
                    if fails > 3:
                        self.fail_accuracy = True
                    if (self.error_RMSE[epoch - 1] - self.error_RMSE[epoch]) < 0.000001:
                            raise Expections.ErrorHasTooLittleChanges
                if self.fail_accuracy == True:
                        raise Expections.AccurancyDecreasedToManyTimes
                if self.error_RMSE[epoch] < 0.005:
                        raise Expections.ErrorWasEnoughSmall

            except Expections.AccurancyDecreasedToManyTimes:
                break
            except Expections.ErrorWasEnoughSmall:
                break
            except Expections.ErrorHasTooLittleChanges:
                break
            # Display errors for network in epoch.
            if epoch % 1 == 0:
                print(f"Epoch {epoch}  |  Error ME {round(ME,5)}    |   Error RMSE {round(RMSE,5)}   |   Error MSE {round(MSE,5)} ")

        # Display plots.
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
        plt.xlabel("Epochs")
        ax1.plot(self.error_RMSE, color='purple', label='RMSE error')
        ax1.plot(self.errors_ME, color='green', label=' ME error')
        ax1.plot(self.error_MSE, color='blue', label=' MSE error')
        ax1.grid(True)
        ax1.legend()

        plt.xlabel("Epochs")
        ax2.plot(self.accuracy_validation_list, color='blue', label='Accuracy (%)')
        ax2.grid(True)
        ax2.legend()
        plt.show()
        # Neural network accuracy plot.
        self.result_accuracy_on_test_data = self.accuracy(self.input_layer_test, self.output_layer_test)
        print(f"Neural network is good at {round(self.result_accuracy_on_test_data,5)} percent")

    def calculate_output(self, instance):
        neurons_in_layers = []
        input = instance

        for j in range(self.number_of_hidden_layers + 1):
            synapses_sum = self.activation_function(np.dot(input, self.init_weights[j]))
            neurons_in_layers.append(synapses_sum)
            input = neurons_in_layers[j]

        return closest_value.closest(self.unique_list_of_targets_class, neurons_in_layers[-1])
