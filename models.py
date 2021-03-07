import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        
        return nn.DotProduct(self.w,x) #just return the dot product 

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        
        dp = nn.as_scalar(self.run(x))
        if dp >= 0.0: #if dot product postive then return 1 
            return 1
        if dp <= 0.0 :#if dot product negative then return -1
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        #trying the sample code and observing x and y's value. 
        #batch_size = 1
        #for x, y in dataset.iterate_once(batch_size):
            #print(x)
            #print(y)
            #break
        flag = True 
        while  flag:
            flag = 0 
            data = dataset.iterate_once(1)
            for cordinates in data:
                #print(x, self.w.data, self.get_prediction(cordinates[0]), nn.as_scalar(cordinates[1])
                #check if the output label given matches the value predicted
                if nn.as_scalar(cordinates[1]) != self.get_prediction(cordinates[0]): 
                    flag += 1
                    #weights are being updated 
                    self.w.update( cordinates[0], nn.as_scalar(cordinates[1]))
                 #loop over the dataset until training accuracy is achieved. If it achieved, terminate the loop
            if flag == 0:
                break 
                
            
                
                
            

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #defining parameters for the model
        #hidden layers size 
        hls = 200 
        #batch size
        self.batch_size = 1
        self.w0 = nn.Parameter(1,hls)
        self.b0 = nn.Parameter(1,hls)
        self.w1 = nn.Parameter(hls,100)
        self.b1 = nn.Parameter(1,100)
        self.w2 = nn.Parameter(100,1)
        self.b2 = nn.Parameter(1,1)
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #converting linear values into non linear using relu and adding bias 
        First_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(First_layer, self.w1), self.b1))
        Third_layer = nn.AddBias(nn.Linear(second_layer, self.w2), self.b2)
        #now return the last layer 
        return Third_layer
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        
        while True:
            for coordinates in dataset.iterate_once(self.batch_size):
            #gathering gradient values for claculation of the loss
                lossvalue = self.get_loss(coordinates[0],coordinates[1])
                gradients = nn.gradients(lossvalue, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                loss_scalar = nn.as_scalar(lossvalue)
                #gradient based updating 
                for k in range(len([self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])):
                #I tried assigning variable to the below list but did not work out. 
                    [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2][k].update(gradients[k], -0.001)#learning rate is 0.001
                X_loss = dataset.x
                Y_loss = dataset.y
                x_loss = nn.Constant(X_loss) 
                yloss = nn.Constant(Y_loss)
                
                if nn.as_scalar(self.get_loss(x_loss,yloss)) < 0.02:
                    return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #hidden layer size
        
        hls = 200
        #batch size assignment
        self.batch_size = 5
        self.w0 = nn.Parameter(784,hls) #28*28 digits size
        self.b0 = nn.Parameter(1,hls)
        self.w1 = nn.Parameter(hls,hls)
        self.b1 = nn.Parameter(1,hls)
        self.w2 = nn.Parameter(hls,10) #output vector size is 10 
        self.b2 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #doing the same thing as above. Converting the linearity and then adding bias
        First_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(First_layer, self.w1), self.b1))
        Third_layer = nn.AddBias(nn.Linear(second_layer, self.w2), self.b2)
        return Third_layer
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #using softmax loss function
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for coordinates in dataset.iterate_once(self.batch_size):
                #gradient based updatinng
                gradients = nn.gradients(self.get_loss(coordinates[0],coordinates[1]), [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                
                self.w0.update(gradients[0],-0.005)
                self.b0.update(gradients[1],-0.005) 
                self.w1.update(gradients[2],-0.005)   #keeping the learning rate 0.005
                self.b1.update(gradients[3],-0.005)
                self.w2.update(gradients[4],-0.005)
                self.b2.update(gradients[5],-0.005)
            #if the validation set accuracy is met then terminate the loop
            if dataset.get_validation_accuracy() > 0.971:
                break 
                
        

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dim = 5
        self.batch_size = 2
        hls = 400
        self.w0 = nn.Parameter(47,hls) 
        self.b0 = nn.Parameter(1,hls)
        self.w1 = nn.Parameter(hls,hls)
        self.b1 = nn.Parameter(1,hls)
        self.w2 = nn.Parameter(hls,self.dim) #output vector size is 10 
        self.b2 = nn.Parameter(1,self.dim)

                   


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #replace the computational form of linearity to Add form and the result function will be non linear
        predicted_y = nn.AddBias(nn.Linear(xs[0], self.w0),self.b0)
        

        for x in xs:
            xm = nn.Add(nn.Linear(predicted_y, self.w1), nn.Linear(x, self.w0))
            predicted_y = nn.ReLU(nn.AddBias(xm, self.b1))
              

        predicted_y = nn.Linear(predicted_y, self.w2)
        return predicted_y

        
        

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #the logic for this training function will be the same as of question 3
        while True:
            for coordinates in dataset.iterate_once(self.batch_size):
                #gradient based updatinng
                gradients = nn.gradients(self.get_loss(coordinates[0],coordinates[1]), [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                
                self.w0.update(gradients[0],-0.005)
                self.b0.update(gradients[1],-0.005) 
                self.w1.update(gradients[2],-0.005)   #keeping the learning rate 0.005
                self.b1.update(gradients[3],-0.005)
                self.w2.update(gradients[4],-0.005)
                self.b2.update(gradients[5],-0.005)
        
            #we are given that if the accuracy for validation test reaches 81% termimate te loop
            print(dataset.get_validation_accuracy())
            if dataset.get_validation_accuracy() >= 0.85:
                return
        
