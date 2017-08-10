#include <time.h>
#include "perceptron.h"

Perceptron::Perceptron(int inputNumber, int function) {
  srand((unsigned) time(0));  //  Seed with current time

  //  Populate the weight vector with values between -0.5 and 0.5
  for (int i = 0; i < inputNumber; ++i) {
    weightVector.push_back((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5);
    inputVector.resize(inputNumber);
    activationFunc = function;
  }
}

void Perceptron::inputAt(int inputPos, float inputValue) {
  inputVector[inputPos] = inputValue;
}

float Perceptron::calculateNet() {
  float action = 0.0;

  //  Action potential (net response) of the input pattern
  for (int i = 0; i < inputVector.size(); ++i)
    action += inputVector.at(i) * weightVector.at(i);

  switch (activationFunc) {
    case THRESHOLD:  //  simple RELU-esque
      if (action >= 0)
        action = 1;
      else
        action = 0;
      break;
    case SIGMOID:
      action = 1.0 / (1.0 + exp(-action));
      break;
    case HYPERBOLIC_TANGENT:
      action = (exp(2 * action) - 1) / (exp(2 * action) + 1);
      break;
  }

  return action;
}

void Perceptron::adjustWeights(float learningRate, float output, float target) {
  for (int i = 0; i < inputVector.size(); ++i)
    weightVector.at(i) = weightVector.at(i) + learningRate * (target - output) * inputVector.at(i);
}

float Perceptron::recall(float red, float green, float blue) {
  inputVector.at(0) = red;
  inputVector.at(1) = green;
  inputVector.at(2) = blue;

  return calculateNet();
}

void Perceptron::printWeightVec() {
  for (std::vector<float>::const_iterator i = weightVector.begin(); i != weightVector.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;
}

void Perceptron::printInputVec() {
  for (std::vector<float>::const_iterator i = inputVector.begin(); i != inputVector.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;
}
