#include "mlp.h"

// MLP constructor
MLP::MLP(int hL, int hN) {
  // Initialize the filereader
  reader = new FileReader();
  outputN = 10;  //  10 possible outcomes
  hiddenL = hL;
  hiddenN = hN;

  reader = new FileReader();

  // Read the first image to see what kind of input will our not have
  inputN = reader -> getBitmapDimensions();
  if (inputN == -1) {
    printf("There was an error detecting img0.bmp\n\r");
    return;
  }

  // Allocate memory for the weights
  weights.reserve(inputN * hiddenN + (hiddenN * hiddenN * (hiddenL - 1)) + hiddenN * outputN);

  // Resize neuron vectors
  inputNeurons.resize(inputN);
  hiddenNeurons.resize(hiddenL * hiddenN);
  outputNeurons.resize(outputN);

  // Initialize the weights of the first hidden layer and input layer
  for (int i = 0; i < inputN * hiddenN; ++i)
    weights.push_back((static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + static_cast<float>(1)) - 0.5));

  // If there are more than one hidden layer, initialize their weights
  for (int a = 1; a < hiddenL; ++a) {
    for (int j = 0; j < hiddenN * hiddenN; ++j)
      weights.push_back((static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + static_cast<float>(1)) - 0.5));
  }
  
  // Initialize the weights of the output layer
  for (int b = 0; b < hiddenN * outputN; ++b)
    weights.push_back((static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + static_cast<float>(1)) - 0.5));
}

// MLP Destructor
MLP::~MLP() {
  delete reader;

  weights.clear();
  inputNeurons.clear();
  hiddenNeurons.clear();
  outputNeurons.clear();
}

// Assign values to the input neurons
bool MLP::populateInput(int fileNum) {
  char* data;
  // Read in image data
  if (!reader -> readBitmap(fileNum))
    return false;

  // Grab image data
  data = reader -> getImgData();

  // Data is stored in every fourth byte of the image
  // Need to remove and compact the useful data
  int n = static_cast<int>(ceil(static_cast<float>(reader -> returnSize()) / 4));
  signed char B[n];

  // Storing in reverse because bmp images store from bottom up.
  for (int i = n - 1; i >= 0; --i)
    B[i] = data[4 * i];

  // Assign value to input neurons
  for (int i = 0; i < inputN; ++i) {
    if (B[i / 8] & (0x01 << (7 - (i % 8))))
      inputNeurons.at(i) = 0.0;  //  If the pixel is white, turn neuron off
    else
      inputNeurons.at(i) = 1.0;  //  Otherwise, the neuron is on
  }
  return true;
}

void MLP::calculateNetwork() {
  // Propagate towards the hidden layers
  for (int hidden = 0; hidden < hiddenN; ++hidden) {
    hiddenAt(1, hidden) = 0;
    for (int input = 0; input < inputN; ++input)
      hiddenAt(1, hidden) += inputNeurons.at(input) * inputToHidden(input, hidden);
    // Pass the sum to the activation function
    hiddenAt(1, hidden) = sigmoid(hiddenAt(1, hidden));
  }

  // If there is more than one hidden layer
  for (int i = 2; i <= hiddenL; ++i) {
    // For each layer, calculate the value
    for (int j = 0; j < hiddenN; ++j) {
      hiddenAt(i, j) = 0;
      for (int k = 0; k < hiddenN; ++k)
        hiddenAt(i, j) += hiddenAt(i - 1, k) * hiddenToHidden(i, k, j);
      // Pass calculated value through the activation function
      hiddenAt(i, j) = sigmoid(hiddenAt(i, j));
    }
  }

  // Hidden to output
  // htoc = hidden to output counter
  for (int i = 0; i < outputN; ++i) {
    outputNeurons.at(i) = 0;
    for (int j = 0; j < hiddenN; ++j)
      outputNeurons.at(i) += hiddenAt(hiddenL, j) * hiddenToOutput(j, i);
    // Pass calculated value to the activation function
    outputNeurons.at(i) = sigmoid(outputNeurons.at(i));
  }
}

// trains the network through back-propagation
bool MLP::trainNetwork(float teachingStep, float lmse, float momentum, int trainingFiles) {
  float mse = 999.0;
  int trainingCounter = 0;
  int epochs = 1;
  float error = 0.0;
  // Delta of the output layer
  float* oDelta = reinterpret_cast<float*>(malloc(sizeof(float) * outputN));
  // Deltas of the hidden layers
  float* hDelta = reinterpret_cast<float*>(malloc(sizeof(float) * hiddenN * hiddenL));
  // Buffer of temporary weights
  std::vector<float> tempWeights = weights;
  // Buffer of to keep the previous weights for the momentum
  std::vector<float> preWeights = weights;

  // Grab image goal i.e. label
  printf("Grabbing the image goals...\n\r");
  int* goals = reader -> getImgGoals();
  
  // Goal number must equal number of training images
  if ((*goals) != trainingFiles) {
    printf("The number of goals does not equal to the number of training images\n\r");
    return false;
  }

  while (fabs(mse - lmse) > 0.0001) {
    // Reset the mean squared error for each epoch
    mse = 0.0;

    // For each file
    while (trainingCounter < trainingFiles) {
      // Populate the input neurons
      if (!populateInput(trainingCounter)) {
        printf("There was an error reading in the image data\n\r");
        return false;
      }

      // Calculate the network
      calculateNetwork();

      // Grab the specific goal value
      int target = *(goals + trainingCounter + 1);

      // Propagate the error backwards through the network
      // through the back-propagation algorithm.
      // First, calculate the delta of the output layer and accumalted error.
      for (int i = 0; i < outputN; ++i) {
        if (i != target) {
          outputDeltaAt(i) = (0.0 - outputNeurons[i]) * dersigmoid(outputNeurons[i]);
          error += (0.0 - outputNeurons[i]) * (0.0 - outputNeurons[i]);
        } else {
          outputDeltaAt(i) = (1.0 - outputNeurons[i]) * dersigmoid(outputNeurons[i]);
          error += (1.0 - outputNeurons[i]) * (1.0 - outputNeurons[i]);
        }
      }

      // Let's start propagating backwards through the first hidden layer
      for (int i = 0; i < hiddenN; ++i) {
        hiddenDeltaAt(hiddenL, i) = 0;  //  Zero out values from previous iteration
        // Sum the deltas for all connections for this neuron
        for (int j = 0; j < outputN; ++j)
          hiddenDeltaAt(hiddenL, i) += outputDeltaAt(j) * hiddenToOutput(i, j);
        hiddenDeltaAt(hiddenL, i) *= dersigmoid(hiddenAt(hiddenL, i));
      }

      // Continue through hidden layers if there are more than one hidden layer
      for (int i = hiddenL - 1; i > 0; --i) {
        for (int j = 0; j < hiddenN; ++j) {
          hiddenDeltaAt(i, j) = 0;
          for (int k = 0; k < hiddenN; ++k)
            hiddenDeltaAt(i, j) += hiddenDeltaAt(i + 1, k) * hiddenToHidden(i + 1, j, k);
          hiddenDeltaAt(i, j) *= dersigmoid(hiddenAt(i, j));
        }
      }

      // Weights modification
      tempWeights = weights;  //  Keep the previous weights, we will need them later

      // Hidden to Input weights
      for (int i = 0; i < inputN; ++i) {
        for (int j = 0; j < hiddenN; ++j)
          inputToHidden(i, j) += momentum * (inputToHidden(i, j) - prev_inputToHidden(i, j)) +
                                  teachingStep * hiddenDeltaAt(1, j) * inputNeurons[i];
      }

      // Hidden to Hidden weights if there are more are more than one hidden layer
      for (int i = 2; i <= hiddenL; ++i) {
        for (int j = 0; j < hiddenN; ++j) {
          for (int k = 0; k < hiddenN; ++k)
            hiddenToHidden(i, j, k) += momentum * (hiddenToHidden(i, j, k) - prev_hiddenToHidden(i, j, k)) +
                                        teachingStep * hiddenDeltaAt(i, k) * hiddenAt(i - 1, j);
        }
      }

      // Hidden to Output weights
      for (int i = 0; i < outputN; ++i) {
        for (int j = 0; j < hiddenN; ++j)
          hiddenToOutput(j, i) += momentum * (hiddenToOutput(j, i) - prev_hiddenToOutput(j, i)) +
                                  teachingStep * outputDeltaAt(i) * hiddenAt(hiddenL, j);
      }

      preWeights = tempWeights;

      // Sum the total mean squared error
      mse += error / (outputN + 1);
      // Reset error for next iteration
      error = 0;

      trainingCounter++;
    }

    // Reset Counter
    trainingCounter = 0;

    // Prompt user if they want to continue training
    char reply;
    if ((epochs % 10000) == 0) {
      printf("We are currently in epoch %d. Would you like to continue training? (N for no, any other key to continue)\n\r", epochs);
      std::cin >> reply;
    }

    if (reply == 'N')
      break;

    printf("Mean squared error for epcoh %d is %.5f. ", epochs, mse);
    printf("While condition (mse - lmse): %.5f\n\r", fabs(mse - lmse));
    epochs++;
  }

  printf("\n\r");
  return true;
}

void MLP::printInput() {
  for (int i = 0; i < inputN; ++i)
    printf("%f ", inputNeurons[i]);
  printf("\n\r");
}

// This function will allow the user to validate and test the network
void MLP::recallNetwork(int fileNum) {
  // Populate the input neurons
  printf("Populating the input with image %d\n\r", fileNum);
  populateInput(fileNum);
  
  // Calculate the network
  printf("Calculating network\n\r");
  calculateNetwork();

  float winner = -1;
  int index = 0;

  // Find the best fitting output
  for (int i = 0; i < outputN; ++i) {
    if (outputNeurons[i] > winner) {
      winner = outputNeurons[i];
      index = i;
    }
  }

  // Print out results
  printf("The network thinks that image %d represents a \n\r\n\r \t\t----->| %d |<------\t\t \n\r\n\r", fileNum, index);
  // Display the probabilities of each class
  printf("Confidence of the network for image %d: \n\r\
          |0 with %d%% probability | \n\r\
          |1 with %d%% probability | \n\r\
          |2 with %d%% probability | \n\r\
          |3 with %d%% probability | \n\r\
          |4 with %d%% probability | \n\r\
          |5 with %d%% probability | \n\r\
          |6 with %d%% probability | \n\r\
          |7 with %d%% probability | \n\r\
          |8 with %d%% probability | \n\r\
          |9 with %d%% probability | \n\r\n\r", fileNum, static_cast<int>(outputNeurons[0] * 100),
          static_cast<int>(outputNeurons[1] * 100), static_cast<int>(outputNeurons[2] * 100),
          static_cast<int>(outputNeurons[3] * 100), static_cast<int>(outputNeurons[4] * 100),
          static_cast<int>(outputNeurons[5] * 100), static_cast<int>(outputNeurons[6] * 100),
          static_cast<int>(outputNeurons[7] * 100), static_cast<int>(outputNeurons[8] * 100),
          static_cast<int>(outputNeurons[9] * 100));
}
