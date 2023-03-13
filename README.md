# NeuralNetworkConsoleApp

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Code examples and how to use model](#code-examples-and-how-to-use-model)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Features](#features)
* [Inspiration](#inspiration)


## General info 

Neural network model from scratch using only numpy, pandas and matplotlib packages. Updating weights using backpropagation algorithm. 
Pre-prepared datasets that can be used for training (pre-prepared in databases.py file):
* Breast Cancer Wisconsin (Diagnostic) Data Set (https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
* Glass Identification Data Set (https://archive.ics.uci.edu/ml/datasets/glass+identification)
* Wine Data Set (https://archive.ics.uci.edu/ml/datasets/Wine)


## Setup

#### 1. Install Python
Install ```python-3.9.13``` and ```python-pip```. Follow the steps from the below reference document based on your Operating System.
Reference: [https://docs.python-guide.org/starting/installation/](https://docs.python-guide.org/starting/installation/)

#### 2. Setup Numpy, Pandas, Matplotlib
```bash
pip install numpy
pip install pandas
pip install matplotlib
```

#### 3. Clone git repository
```bash
git clone "https://github.com/igness1/NeuralNetworkConsoleApp.git"
```

#### 4. Example training process using this neural network model
Run powershell or cmd console.
The model the model uses the breast cancer database, 
but you can change to another prepared in the databases.py file. 
You can use another database, but you need to prepared/preprocessed data like in the examples provided in the databases.py file.
```bash
cd .\SimpleDjangoBlog\
python .\model_use_example.py
```
## Code examples and how to use model
```python
# Load database.
breast_cancer_input, breast_cancer_output, class_names = databases.breast_cancer_database()
```

Using crossvalidation 5-fold to prepare 5 training and testing sets.
```python
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
```
Define the models and start learning process. 
```python
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
```

## Screenshots of the training process.
Example screenshots.

Training process:  
![image](https://user-images.githubusercontent.com/58557112/224812246-be6d1a77-7083-401f-ab8a-663cee541105.png)

Plots with ME, MSE, RMSE values during learning:  
![image](https://user-images.githubusercontent.com/58557112/224812746-4a5ae294-cd7a-44b0-b1bd-80ab037c2e62.png)

Accuracy of the model tested on a testing set: 
![image](https://user-images.githubusercontent.com/58557112/224816647-9f2bb28c-004f-4d9e-821c-8a1ed6c9389e.png)
![image](https://user-images.githubusercontent.com/58557112/224816749-d0742efe-2346-4de8-b0e0-b847636af500.png)


## Technologies
* Python - 3.9.13
* Numpy - 1.24.2
* Pandas - 1.5.3
* Matplotlib - 3.7.1

## Features
List of features ready: 
* Define network parameters: input/output training/testing data, learning_rate, 
activation function (available: sigmoid, tahn, unistep, ReLU), momentum hiperparameter, 
number of hidden layers and neurons.
* Train model.
* Make a prediction using trained model on new sample/pattern.

To-do list:
* Refactor code to be less spaghetti code.
* Think how to add different optimalization algorithms.


## Inspiration
My inspiration to create such an app was a willing to learn mathematical aspects related to the operation of neural networks.
