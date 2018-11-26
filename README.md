# Hand_Gesture-
A project that identifies hand gestures from a trained data set.
1. Overview

This project aims to identify hand gestures using Back Propagation techniques of Artificial Neural Networks. The hand gestures chosen for the project are specific to denote the numbers from 1 to 5. These are chosen from a dataset that can denote each of the numerals in a different form for a different input. The number of fingers open, represent the numbers.The simulated neural network identifies each representation of the numeral correctly with maximum accuracy and minimal loss. 
The dataset used as input to the network consists of coloured images representing the numerals from  1 to 5. Further processing of the dataset, to recognise the contour and the convex hull of the hand is done by the network. 

Pillow and NumPy
Pillow is a fork of PIL(Python Image Library). PIL is a free library that adds support for opening,manipulating and many different image file format. Pillow is a newer version of PIL and as been adopted as the replacement for the PIL.  BMP, EPS, GIF, IM, JPEG, MSP, PCX, PNG, PPM, TIFF, WebP, ICO, PSD, PDF are some of the file formats supported by Pillow. “PNG” is the file format used for the  data set in this project.
NumPy is a library that adds support for large multi-dimensional arrays and matrices and a large set of mathematical functions to operate on these arrays. The use of NumPy gives the functionalities similar to that of MATLAB, a numerical computing environment. In this project, the images which are being opened are being stored in the format of Numpy array, imported as np.
im = Image.open(d)
mat = np.array(im)
The input and output neurons initially taken in the form of a list have also been converted into the numpy format.
	X=[]
	Y=[]
Y = np.array(Y)
	X = np.array(X) 


The Neural Network is designed using the Keras Library for deep learning on the Python platform. The network has 5 layers including the input layer and the output layer. Each neuron in these layers processes the input signals using the ReLU (Rectified Linear Unit) activation function. The network identifies the defect points (valley between two fingers) of the hand as well as the fingertips and maps the same to get the convex hull of the hand. This helps it learn and further identify the hand gesture and map it accurately. The network adjusts the weights between neurons of subsequent layers to fit to the important part of the image it has assembled. 

Progress 
The dataset for the project has been downloaded and preprocessed. The dataset is used from the Creative Senz3D camera device and consisting of over 1200 images that constitute the input to the network. Since this data was not fit to be directly input to the network it was segmented in a suitable form and manually written into a Python Dictionary. Each element of the dictionary has the name of the image as key and result of the image as it’s value. After this, as the images were read they were converted into NumPy arrays and coloured images to Grayscale images and collected as the final input to the network. After this phase, the neural network was trained using 85% of the dataset as the training data. The preprocessing phase of the testing data is also completed giving the data in a readable, ready to use input format.


2.Related Works

Hand Gesture recognition systems find their applications in various fields such as Sign Language Interpretation, Natural Interface Control, Robotic Control etc. This concept has been researched, implemented and optimised for these various operations. This is owing to the fact that Neural networks are flexible in a changing environment. Although neural networks may take some time to learn a sudden drastic change, they are excellent at adapting to constantly changing information. Neural networks can build informative models where more conventional approaches fail. Because neural networks can handle very complex interactions they can easily model data which is too difficult to model with traditional approaches such as inferential statistics or programming logic. Performance of neural networks is at least as good as classical statistical modeling, and better on most problems. The neural networks build models that are more reflective of the structure of the data in significantly less time.

Perceptron Model Approach was used by Paul Morton [1] in his report on ‘Hand Gesture Recognition using Neural Networks’ published for Armstrong Laboratories. The project was used for motor function sensing among the candidates. A multilayer perceptron model was used which consisted of 12 input, 15 hidden layer nodes and 25 output nodes. The approach used 12 inputs collected from the glove with the attached Absolute Position and Orientation Sensor. These inputs were the angular displacement of the wrist and fingers collected by the sensors. Pilot data from 3 candidates were used to train and fine-tune the multilayer perceptron network and explore alternate paradigms. Once this training phase was completed, the network was applied on the gesture collected by 10 other candidates. Results were obtained in terms of percentage of total recognition activity and nature of the errors made. The lowest recognition percentage obtained was 79% when tested by person with no motor inactivity owing to the issue of testing on small dataset. Nevertheless the average recognition rate for the candidates with motor impairments was 86.28% which was slightly less than the recognition rate for able-bodied candidates (92.28%). 
Contributions by Bhushan Bhokse and Dr. A. R. Karwankar [2] in their paper on ‘Hand Gesture Recognition using Neural Networks’ discusses the feed forward network on a set of images that were scaled to binary. Data preprocessing included the stages of ‘zooming in’ of the  binary images which basically means to identify the part of the picture that contains the hand. Methods like Background Subtraction, Segmentation and Thinning were used before zooming in owing to clear images with no noise or ambiguity of hand gesture. There were three layers in the neural network used, namely the input, hidden and output layer. The number of neurons used in the input layer depended on the number of attributes of the image. The dataset used comprised of 5 gestures in one hand that would be recognised by the network. The output of the network classified between 5 kinds of signs. As a consequence, it necessarily included 5 output neurons, so that the whole system provided five output values, which was considered as a 5x1 vector. The paper also discussed some of the issues involved in this process and provided a detailed description of the issue with zooming of the image. After zooming, the next issue that would have to resolved was the standard resizing of the picture to fit the frame. The authors conclude that if the neural network method is the most efficient solution when considering the rate of errors, another method based on attributing weights to each column of the input picture, is quite easier to develop and manipulate, and has also been proven very efficient.
 
3. Experimental Design
  
Basically an  artificial neural network is a group of nodes interconnected so as to simulate the functions and operations of the human brain. A typical neural network is fed large amounts of data for the training phase. Training involves providing the input and telling the network what the output should be. The network then learns accordingly. The nodes of the artificial neural network is the artificial neuron. The connections between these neurons typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Artificial neurons may have a threshold value set,such that, if the aggregate of signals is equal to the threshold or crosses it,the signal is sent to the next neuron, i.e the neuron is fired. Typically, artificial neurons are organized in layers, each of which performs an operation on their inputs.

In our project we are trying to simulate a neural network which recognizes the hand gestures which are numbers from 1 to 5. The network is a feed forward network consisting of an input layer, 4 hidden layers and an output layer. Input layer consists of 40 neurons with 70 input features. Each of the hidden layer consists of 40 neurons and a single neuron in the output layer.
The network is trained using back propagation technique. The backpropagation algorithm is used to find a local minimum of the error function. The network is initialized with randomly chosen weights. The gradient of the error function is computed and used to correct the initial weights. Our task is to compute this gradient recursively.

Dataset
Our database for hand gesture recognition (HGR) contains the gestures indicating numerals 0 to 5 using a single hand. The data is modified to user and project requirements. The database was developed as a part of the hand detection and pose estimation project, supported by the Polish Ministry of Science and Higher Education under research grant no. IP2011 023071 from the Science Budget 2012-2013.
      

	Each of the image is read as a coloured (original) image which is converted to a grayscale image (Skin mask) or Visual and stored in a matrix. The matrix is then flattened to 1 dimension and appended to an array. This process is repeated for all the features whose values are stored in a separate dictionary. Then comes the process of constructing, training and evaluating the network. The entire training and testing of the network is done using an application called Keras. 

Keras: The Python Deep Learning Library
Keras is a high level neural  networks API, written in python and capable of running on top of TensorFlow, CNTK and theano. It supports both convolutional networks and recurrent networks, as well as the combination of the two.
Keras contains numerous implementations of commonly used neural networks building blocks such as layers, objectives, activation functions, optimizers and a host of tools to make working with images and text data easier.
In this project Keras will use TensorFlow as its tensor manipulation library.

Below is an overview of the 5 steps in the neural network model life-cycle in Keras that we are going to look at.
Define Network.
Compile Network.
Fit Network.
Evaluate Network.
Make Predictions.



5 Step Life-Cycle for Neural Network Models in Keras











Step 1: Define Model
The first step is to define the neural network. Neural networks are defined in Keras as a sequence of layers. The container for these layers is the Sequential class. The first step is to create an instance of the Sequential class. Then we can create layers and add them in the order that they should be connected.

model = Sequential()
model.add(Dense(10, input_dim=70))
model.add(Dense(5))

The above defines  a multi layer perceptron network which is defined by adding 10 input neurons each with 70 features and a hidden layer with 5 neurons. It is very necessary to define the activation function for each of the layers in network which does the transformation of data from input to prediction.It can be achieved by using the following statements

model = Sequential()
model.add(Dense(10, input_dim=70))
model.add(Activation(‘relu’))
model.add(Dense(5))
model.add(Activation(‘relu’))

Here we are using an activation function called rectified linear unit simply called as relu.It is defined as

A(x) = max(0,x)

The ReLu function is as shown above. It gives an output x if x is positive and 0 otherwise.

ReLu is nonlinear in nature. And combinations of ReLu are also non linear! ( in fact it is a good approximator. Any function can be approximated with combinations of ReLu). This means we can stack layers. It is not bound though. The range of ReLu is [0, inf). This means it can blow up the activation. The choice of activation function is most important for the output layer as it will define the format that predictions will take.

Step 2: Compile Network
Once we have defined our network, we must compile it.Compilation is an efficiency step. It transforms the simple sequence of layers that we defined into a highly efficient series of matrix transforms in a format intended to be executed on your GPU or CPU.It is always better to think of compilation as a precompute step for our network. Compilation is always required after defining a model. This includes both-before training it, using an optimization scheme as well as loading a set of pre-trained weights from a saved file. The reason is that the compilation step prepares an efficient representation of the network that is also required to make predictions. Compilation requires a number of parameters to be specified, specifically tailored to training our network. Specifically the optimization algorithm is used to train the network and the loss function is used to evaluate the network that is minimized by the optimization algorithm.

model.compile(optimizer='adam', loss='binary_crossentropy')

Here we are using the stochastic gradient descent(sgd) technique as the optimizer and the mean squared error(mse) for defining the loss function.

Step 3: Fit Network
Once the network has been compiled, we need to fit the network, which means adapt the weights on a training dataset. Fitting the network means where the training data has to be specified. In the training data, the input patterns X is represented in terms of a matrix with an array of matching output patterns Y. The network is trained using the back propagation algorithm and optimized according to the optimization algorithm and loss function specified when compiling the model.
Back propagation is one of the methods used to calculate the gradient which is used in calculating the weights to be used in the network. The back propagation also requires that the network be trained for a specified number of epochs to the training dataset. Each epoch can be partitioned into groups of input-output pattern pairs called batches. These batches will define the number of patterns that the network is exposed to before updating the weights within a epoch.
Fitting the network is done as follows:

history=model.fit(X,Y, batch_size=5, epoches=50)

Once the model is fit, a history object is returned that provides the summary of the performance of the model during training.  This includes both the loss and any additional metrics specified when compiling the model, recorded each epoch.

Step 4: Evaluate network
After the training of the network, the network has to be evaluated. We can evaluate the  performance of the network on a separate dataset, unseen by the network during testing. This will provide an estimate of the performance of the network at making predictions for unseen data in future.
The model evaluates the loss across all of the test patterns and any other metric which were specified when the model was compiled, like classification accuracy. A list of evaluation metrics are returned.
For a model compiled with accuracy metric, it can be evaluated on a new dataset as follows:

Loss, accuracy=model.evaluate(X,Y)

Step 5: Make predictions
Once the performance of the model is satisfied after the fit, we can use the model to make predictions on new data. This is as easy as calling the predict() on the model with an array of new input patterns.

predictions=model.predict(x)

The predictions will be returned in the format in the format provided by the output layer of the network.


