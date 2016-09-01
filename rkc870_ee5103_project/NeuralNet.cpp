//
//  NeuralNet.cpp
//  rkc870_ee5103_project


#include <stdio.h>
#include "NeuralNet.h"

NeuralNet::NeuralNet(){}

//initializes all values to zero
void NeuralNet::initialize(double weights[][arraySize], double values[], double expectedValues[], double thresholds[]) {
    //nodes 1 and 2 are input nodes, 3 and 4 are hidden nodes, and node 5 is the output node
    for (int i = 0; i <= nodes; i++) {
        values[i] = 0.0;
        expectedValues[i] = 0.0;
        thresholds[i] = 0.0;
        for (int j = 0; j <= nodes; j++) {
            weights[i][j] = 0.0;
        }
    }
}

//initializes node connections with random weights from zero to two
void NeuralNet::connectNodes(double weights[][arraySize], double thresholds[]) {
    for (int i = 0; i <= nodes; i++) {
        for (int j = 0; j <= nodes; j++) {
            weights[i][j] = (rand() % 200)/100.0;
        }
    }
    //thresholds are not needed for the input nodes
    //initialize all other thresholds to completely random values
    thresholds[3] = rand()/(double) rand();
    thresholds[4] = rand()/(double) rand();
    thresholds[5] = rand()/(double) rand();
    
    //print out weights of all node connections (input to hidden and then hidden to output)
    std::cout << "weight from node 1 to node 3 is: " << weights[1][3] << "\n";
    std::cout << "weight from node 1 to node 4 is: " << weights[1][4] << "\n";
    std::cout << "weight from node 2 to node 3 is: " << weights[2][3] << "\n";
    std::cout << "weight fromm node 2 to node 4 is: " << weights[2][4] << "\n";
    std::cout << "weight from node 3 to node 5 is: " << weights[3][5] << "\n";
    std::cout << "weight from node 4 to node 5 is: " << weights[4][5] << "\n";
    //print thresholds for hidden nodes and output nodes
    std::cout << "threshold for node 3 is: " << thresholds[3] << "\n";
    std::cout << "threshold for node 4 is: " << thresholds[4] << "\n";
    std::cout << "threshold for node 5 is: " << thresholds[5] << "\n";
}

//generates training data
void NeuralNet::trainingSample(double values[], double expectedValues[]) {
    //counter must be static so that it retains its value through all iterations
    static int counter = 0;
    
    //this is essentially a truth table for the XOR gate
    switch (counter % 4) {
            
        case 0:
            values[1] = 0;
            values[2] = 0;
            expectedValues[5] = 0;
            break;
        case 1:
            values[1] = 0;
            values[2] = 1;
            expectedValues[5] = 1;
            break;
        case 2:
            values[1] = 1;
            values[2] = 0;
            expectedValues[5] = 1;
            break;
        case 3:
            values[1] = 1;
            values[2] = 1;
            expectedValues[5] = 0;
            break;
            
    }
    
    counter++;
}

//starts to run the network- processes the hidden and output nodes
void NeuralNet::runNetwork(double weights[][arraySize], double values[], double thresholds[]) {
    
    
    //iterate through each hidden node
    for (int i = 1 + inputNodes; i <= 1 + hiddenNodes + inputNodes; i++) {
        //iterate through each input node
        double weightedInputs = 0.0; //declaration must be inside for loop so that each node has a weighted input
        for (int j = 1; j <= 1 + inputNodes; j++) {
            weightedInputs += weights[j][i] * values[j]; //add weights value product at each hidden node
        }
        weightedInputs -= thresholds[i]; //subtract thresholds once hidden nodes have sum of weighted inputs
        values[i] = 1.0/(1.0 + pow(e, -1 * weightedInputs)); //apply sigmoid function to hidden nodes
    }
    
    //iterate through each output node and do the same thing as above for hidden nodes
    for (int k = 1 + inputNodes + hiddenNodes; k <= nodes; k++) {
        //iterate through each hidden node
        double weightedInputs = 0.0;
        for (int l = 1 + inputNodes; l <= 1 + inputNodes + hiddenNodes; l++) {
            weightedInputs += weights[l][k] * values[l];
        }
        weightedInputs -= thresholds[k];
        values[k] = 1.0/(1.0 + pow(e, -1 * weightedInputs));
    }
}

//updates weights using error gradient descent back propagation technique
double NeuralNet::updateWeights(double weights[][arraySize], double values[], double expectedValues[], double thresholds[]) {
    double sumOfSquaredErrors = 0.0;
    
    //iterate through output nodes
    for (int i = 1 + inputNodes + hiddenNodes; i <= nodes; i++) {
        
        double absoluteError = expectedValues[i] - values[i]; //calculate error for each output node
        sumOfSquaredErrors += pow(absoluteError, 2); //square the absolute error
        double outputErrorGradient = values[i] * (1.0 - values[i]) * absoluteError; //calculate error gradient
        
        //iterate through hidden nodes
        for (int j = 1 + inputNodes; j <= 1 + inputNodes + hiddenNodes; j++) {
            
            double delta = learningRate * values[j] * outputErrorGradient;//calculate change in weights
            weights[j][i] += delta; //adjust weights for hidden node connections
            double hiddenErrorGradient = values[j] * (1 - values[j]) * outputErrorGradient * weights[j][i];
            
            //iterate through all nodes
            for (int k = 1; k <= nodes; k++) {
                
                double delta = learningRate * values[k] * hiddenErrorGradient; //calculate change in weights for all nodes
                weights[k][j] += delta; //adjust weights for all nodes
                
            }
            
            double thresholdDelta = -1 * learningRate * hiddenErrorGradient;
            thresholds[j] += thresholdDelta; //adjust thresholds for hidden nodes
            
        }
        
        //update weights for output node connections
        double delta = -1 * learningRate * outputErrorGradient;
        thresholds[i] += delta;
    }
    
    return sumOfSquaredErrors;
    
}

//display the two inputs, single output, and sum of squared errors
void NeuralNet::displayNetwork(double values[], double sumOfSquaredErrors) {
    static int counter = 0;
    
    //print divider every 4th line and after display values mentioned above
    if (counter % 4 == 0) std::cout << "\n\n********************************************\n\n";
    std::cout << "input 1 = " << values[1] << " | ";
    std::cout << "input 2 = " << values[2] << " | ";
    std::cout << "output = " << values[5] << " | ";
    std::cout << "sum of squared errors = " << sumOfSquaredErrors << "\n";
    counter++;
}

//learn mode member function
void NeuralNet::learnMode() {
    /*double weights[arraySize][arraySize];
    double values[arraySize];
    double expectedValues[arraySize];
    double thresholds[arraySize];
    int counter = 0;
    double sumOfSquaredErrors = 0;*/
    
    initialize(weights, values, expectedValues, thresholds);
    connectNodes(weights, thresholds);
    
    //main learning loop of the program
    while (counter < maxIterations) {
        trainingSample(values, expectedValues); //generate training sample data
        runNetwork(weights, values, thresholds); //run the example through the network
        sumOfSquaredErrors = updateWeights(weights, values, expectedValues, thresholds); //update all values (learn)
        displayNetwork(values, sumOfSquaredErrors); //display the network
        counter++; //increment counter until max iterations is reached
    }
}

//input mode
void NeuralNet::inputMode(int input1, int input2) {
    values[1] = input1;
    values[2] = input2;
    runNetwork(weights, values, thresholds);
    std::cout << "\n\ninput1 = " << values[1] << " | ";
    std::cout << "input2 = " << values[2] << "\n\n";
    std::cout << "input1 XOR input2 = " << ((values[5] > 0.5) ? "1" : "0") << "\n";
}

