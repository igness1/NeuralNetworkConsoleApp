from neural_network import neuralNetwork
import databases
import matplotlib.pyplot as plt
import numpy as np
from crossvalidation import cross_validation_5_fold as crossvalidation

# Load database.
breast_cancer_input, breast_cancer_output, class_names = databases.breast_cancer_database()

# Model 1 
model_1 = crossvalidation(breast_cancer_input, breast_cancer_output, set_number=1)
breast_cancer_input_train_model_1, breast_cancer_output_train_model_1 = model_1.get_train_set()
breast_cancer_input_test_model_1, breast_cancer_output_test_model_1 = model_1.get_test_set()
# Model 2
model_2 = crossvalidation(breast_cancer_input, breast_cancer_output, set_number=2)
bc_input_train_model_2, bc_output_train_model_2 = model_2.get_train_set()
bc_input_test_model_2, bc_output_test_model_2 = model_2.get_test_set()
# Model 3
model_3 = crossvalidation(breast_cancer_input, breast_cancer_output, set_number=3)
bc_input_train_model_3, bc_output_train_model_3 = model_3.get_train_set()
bc_input_test_model_3, bc_output_test_model_3 = model_3.get_test_set()
# Model 4
model_4 = crossvalidation(breast_cancer_input, breast_cancer_output, set_number=4)
bc_input_train_model_4, bc_output_train_model_4 = model_4.get_train_set()
bc_input_test_model_4, bc_output_test_model_4 = model_4.get_test_set()
# Model 5
model_5 = crossvalidation(breast_cancer_input, breast_cancer_output, set_number=5)
bc_input_train_model_5, bc_output_train_model_5 = model_5.get_train_set()
bc_input_test_model_5, bc_output_test_model_5 = model_5.get_test_set()

# Model 1 training process
neural_model_1 = neuralNetwork(breast_cancer_input_train_model_1, breast_cancer_output_train_model_1, breast_cancer_input_test_model_1, breast_cancer_output_test_model_1, class_names, number_of_neurons_in_hidden_layers=[100,100,100], activation_function="sigmoid", learning_rate=0.0015)
neural_model_1.initial_weights()
neural_model_1.train_network(100)
# Model 2 training process
neural_model_2 = neuralNetwork(bc_input_train_model_2, bc_output_train_model_2, bc_input_test_model_2, bc_output_test_model_2, class_names, number_of_neurons_in_hidden_layers=[100,100,100], activation_function="sigmoid", learning_rate=0.0015)
neural_model_2.initial_weights()
neural_model_2.train_network(100)
# Model 3 training process
neural_model_3 = neuralNetwork(bc_input_train_model_3, bc_output_train_model_3, bc_input_test_model_3, bc_output_test_model_3, class_names, number_of_neurons_in_hidden_layers=[100,100,100], activation_function="sigmoid", learning_rate=0.0015)
neural_model_3.initial_weights()
neural_model_3.train_network(100)
# Model 4 training process
neural_model_4 = neuralNetwork(bc_input_train_model_4, bc_output_train_model_4, bc_input_test_model_4, bc_output_test_model_4, class_names, number_of_neurons_in_hidden_layers=[100,100,100], activation_function="sigmoid", learning_rate=0.0015)
neural_model_4.initial_weights()
neural_model_4.train_network(100)
# Model 5 training process
neural_model_5 = neuralNetwork(bc_input_train_model_5, bc_output_train_model_5, bc_input_test_model_5, bc_output_test_model_5, class_names, number_of_neurons_in_hidden_layers=[100,100,100], activation_function="sigmoid", learning_rate=0.0015)
neural_model_5.initial_weights()
neural_model_5.train_network(100)
