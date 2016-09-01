//
//  Main.cpp
//  rkc870_ee5103_project
//
//  Created by A. Train on 4/9/16.
//  Copyright © 2016 A. Train. All rights reserved.
//

#include <stdio.h>
#include "NeuralNet.h"

int main(int argc, const char * argv[]) {
    int input1 = 0;
    int input2 = 0;
    
    std::cout << "Artificial Neural Network!\n\n";
    std::cout << "Ready to learn?\n(press enter)\n\n";
    std::cin.get();
    NeuralNet::NeuralNet ANN = NeuralNet::NeuralNet();
    ANN.NeuralNet::learnMode();
    
    while(true) {
        std::cout << "\nArtificial Neural Net is ready for inputs:\n(press '10' to quit)\n\n";
        std::cin >> input1;
        if (input1 == 10) break;
        std::cin >> input2;
        if (input2 == 10) break;
        ANN.NeuralNet::inputMode(input1, input2);
    }
    
    
    
    return 0;
}