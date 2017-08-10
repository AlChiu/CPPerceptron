#include <math.h>
#include <iostream>
#include <cstdlib>
#include <vector>

enum activationFunctions {THRESHOLD = 1, SIGMOID, HYPERBOLIC_TANGENT};
class Perceptron {
 private:
  std::vector<float> inputVector;  // Vector of perceptron input values
  std::vector<float> weightVector;  // Vector of perceptron weights
  int activationFunc;
 public:
  Perceptron(int inputNumber, int function);  // Constructor
  void inputAt(int inputPos, float inputValue);  // Input population function
  float calculateNet();  // activation function type
  void adjustWeights(float learningRate, float output, float target);  // Change weights based on output target relation
  float recall(float red, float green, float blue);
  void printWeightVec();
  void printInputVec();
};
