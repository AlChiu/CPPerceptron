#include <stdlib.h>
#include <string>

#include "mlp.h"

int main(int argc, const char* argv[]) {
  int number = 5, eleos = sizeof(std::string);

  if (argc != 7) {
    printf("Incorrect number of arguments detected\n\r\n\r");
    printf("Usage is: %s hiddenNeurons(int) hiddenLayers(int) numberOfBitmaps(int) learningRate(float) leastMSE(float) momentum(float) \n\r", argv[0]);
    printf("\n\rPress any key to continue and exit the program\n\r");
    getchar();
    return -1;
  }

  int hiddenNeurons = atoi(argv[1]);
  int hiddenLayers = atoi(argv[2]);
  int noOfBitmaps = atoi(argv[3]);
  float learningRate = atof(argv[4]);
  float leastMSE = atof(argv[5]);
  float momentum = atof(argv[6]);

  MLP* neuralNetwork = new MLP(hiddenLayers, hiddenNeurons);
  neuralNetwork -> populateInput(2);

  if (!(neuralNetwork -> trainNetwork(learningRate, leastMSE, momentum, noOfBitmaps))) {
    printf("There was an error in training ... Quitting\n\r");
    printf("\n\rPress any key to continue and exit the program\n\r");
    getchar();
    return -1;
  }

  while (number != -1) {
    printf("Training is complete, give the number of the bitmap you want to test (0 - %d), or -1 to exit\n\r\n\r", noOfBitmaps - 1);
    std::cin.clear();
    std::cin >> number;

    if (std::cin.good()) {
      if (number != -1)
        neuralNetwork -> recallNetwork(number);
      else
        break;
    } else {
      continue;  // ask again
    }
  }

  printf("\n\rExiting the program\n\r");
  std::cin.ignore();
  return 0;
}
