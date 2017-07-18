#include "perceptron.h"

// Constructor
perceptron::perceptron(float eta, int epochs)
{
	m_epochs = epochs; // Set private epochs to the user selected epochs
	m_eta = eta; // Set private learning rate to the user selected learning rate
}

// Predict(vector<float> X)
// Input: Vector<float> X
// Output: Returns 1 if netInput(X) > 0, otherwise return -1
int perceptron::predict(std::vector<float> X)
{
	return netInput(X) > 0 ? 1 : -1;
}