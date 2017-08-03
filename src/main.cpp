#include <stdio.h>
#include "perceptron.h"

#define CLASS_BLUE 1
#define CLASS_RED 0
#define NO_PATTERNS 13  // Number of inputs we have
#define LEAST_MSE 0.001  // Error threshold
#define LEARNING_RATE 0.01  // Set learning rate

float normalize(float x);
bool streamIsNumeric(char inp[1024]);

int ourInput[] = {
  // Red Green Blue classes
  0, 0, 255, CLASS_BLUE,
  0, 0, 192, CLASS_BLUE,
  243, 80, 59, CLASS_RED,
  255, 0, 77, CLASS_RED,
  77, 93, 190, CLASS_BLUE,
  255, 89, 98, CLASS_RED,
  208, 0, 49, CLASS_RED,
  67, 15, 210, CLASS_BLUE,
  82, 117, 174, CLASS_BLUE,
  168, 42, 89, CLASS_RED,
  248, 80, 68, CLASS_RED,
  128, 80, 255, CLASS_BLUE,
  228, 105, 116, CLASS_RED
};

int main() {
  float output, result;
  int inputCounter;

  Perceptron RedBlue(3, SIGMOID);
  float mse = 999;
  int epochs = 0;

  // Training of the network
  while (fabs(mse - LEAST_MSE) > 0.0001) {
    mse = 0;
    float error = 0;

    // Run through all patterns will equate to one epoch
    for (int i = 0; i < NO_PATTERNS; ++i) {
      for (int j = 0; j < 3; ++j) {
        RedBlue.inputAt(j, normalize((ourInput[inputCounter])));
        inputCounter++;
      }
      // Calculate the activation function for this pattern
      output = RedBlue.calculateNet();
      // Build up total error for this epoch
      error += fabs(ourInput[inputCounter] - output);
      // Adjust weights based on the error
      RedBlue.adjustWeights(LEARNING_RATE, output, ourInput[inputCounter]);
      // Next Pattern
      inputCounter++;
    }

    // Calculate the mean square error of the epoch
    mse = error/NO_PATTERNS;
    printf("The mean square error of %d epoch is %.4f \r\n", epochs, mse);
    epochs++;
  }

  // With training complete, we can ask the user for input
  int R = -1, G = -1, B = -1;
  char inp[1024];

  while (true) {
    while (R < 0 || R > 255) {
      printf("Give a RED value (0-255)\n\r");
      std::cin.getline(inp, 1024);
      if (!streamIsNumeric(inp))
        continue;
      R = atoi(inp);
    }
    while (G < 0 || G > 255) {
      printf("Give a GREEN value (0-255)\n\r");
      std::cin.getline(inp, 1024);
      if (!streamIsNumeric(inp))
        continue;
      G = atoi(inp);
    }
    while (B < 0 || B > 255) {
      printf("Give a BLUE value (0-255)\n\r");
      std::cin.getline(inp, 1024);
      if (!streamIsNumeric(inp))
        continue;
      B = atoi(inp);
    }

    result = RedBlue.recall(normalize(R), normalize(G), normalize(B));
    if (result > 0.5)
      printf("The value you entered belongs to the BLUE CLASS\n\r");
    else
      printf("The value you entered belongs to the RED CLASS\n\r");

    printf("Do you want to continue with trying  to recall values from the perceptron?");
    printf("\n\r Press any key for YES and 'N' for no, to exit the program\n\r");
    std::cin.getline(inp, 1024);

    // Reset for next run
    B = G = R = -1;

    if (inp[0] == 'N')
      break;
  }

  return 0;
}

float normalize(float x) {
  x = (1./255.) * x;
  return x;
}

bool streamIsNumeric(char inp[1024]) {
  for (int i = 0; i < 1024; ++i) {
    // if the the character is a terminator
    if (inp[i] == '\0')
      break;
    // if the character is not a digit, return false
    if (!isdigit(inp[i]))
      return false;
  }

  return true;
}
