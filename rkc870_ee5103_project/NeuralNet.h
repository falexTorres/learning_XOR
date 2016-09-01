//
//  NeuralNet.h
//  rkc870_ee5103_project

#ifndef NeuralNet_h
#define NeuralNet_h
#include <iostream>
#include <cmath>
#include <string>

//declare constants
const int inputNodes = 2;
const int hiddenNodes = 2;
const int outputNodes = 1;
const int nodes = inputNodes + hiddenNodes + outputNodes;
const int arraySize = nodes + 1; //so that there isn't a zeroth node
//double weights[arraySize][arraySize];
//double values[arraySize];
//double expectedValues[arraySize];
//const int maxIterations = 130000;
const int maxIterations = 15000;
const double e = 2.71828;
const double learningRate = 0.2; //keeps network from over generalizing or under generalizing

class NeuralNet {
  public:
    double weights[arraySize][arraySize];
    double values[arraySize];
    double expectedValues[arraySize];
    double thresholds[arraySize];
    int counter = 0;
    double sumOfSquaredErrors = 0;
    NeuralNet();
    void initialize(double[][arraySize], double[], double[], double[]);
    void connectNodes(double[][arraySize], double[]);
    void trainingSample(double[], double[]);
    void runNetwork(double[][arraySize], double[], double[]);
    double updateWeights(double[][arraySize], double[], double[], double[]);
    void displayNetwork(double[], double);
    void learnMode();
    void inputMode(int, int);
};
#endif /* NeuralNet_h */
