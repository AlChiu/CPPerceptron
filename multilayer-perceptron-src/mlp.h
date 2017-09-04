#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "filereader.h"

// Define several function macros to increase code readability and maintenance
/* Macros to help aid in locating the appropriate weights in the weight vector */
#define inputToHidden(inp, hid) weights.at(inputN * hid + inp)
#define hiddenToHidden(toLayer, fromHid, toHid) weights.at(inputN * hiddenN + ((toLayer - 2) * hiddenN * hiddenN) + hiddenN * fromHid + toHid)
#define hiddenToOutput(hid, out) weights.at(inputN * hiddenN + (hiddenL - 1) * hiddenN * hiddenN + hid * outputN + out)
/* Similar to the above macros, but for the previous weights */
#define prev_inputToHidden(inp, hid) preWeights.at(inputN * hid + inp)
#define prev_hiddenToHidden(toLayer, fromHid, toHid) preWeights.at(inputN * hiddenN + ((toLayer - 2) * hiddenN * hiddenN) + hiddenN * fromHid + toHid)
#define prev_hiddenToOutput(hid, out) preWeights.at(inputN * hiddenN + (hiddenL - 1) * hiddenN * hiddenN + hid * outputN + out)
/* Macro to find the appropriate hidden neuron */
#define hiddenAt(layer, hid) hiddenNeurons[(layer - 1) * hiddenN + hid]
/* Macros to find the appropriate neuron's delta */
#define outputDeltaAt(out) (*(oDelta + out))
#define hiddenDeltaAt(layer, hid) (*(hDelta + (layer - 1) * hiddenN + hid))
/* Macros to define the sigmoid activation function */
#define sigmoid(value) (1 / (1 + exp(-value)));
#define dersigmoid(value) (value * (1 - value))

class MLP {
  private:
    std::vector<float> inputNeurons;
    std::vector<float> hiddenNeurons;
    std::vector<float> outputNeurons;
    std::vector<float> weights;
    FileReader* reader;
    int inputN, outputN, hiddenN, hiddenL;
  public:
    MLP(int hiddenL, int hiddenN);
    ~MLP();
    bool populateInput(int fileNum);
    void calculateNetwork();
    bool trainNetwork(float teachingStep, float lmse, float momentum, int trainingFiles);
    void printInput();
    void recallNetwork(int fileNum);
};
