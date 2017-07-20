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

// netInput(vector<float> X)
// Input: Vector<float> X
// Output: Returns a float that is a sum of all training input * weights + bias
// Example: (x1*w1 + x2*w2 + ... + xn*wn) + bias
float perceptron::netInput(std::vector<float> X)
{
	// Sum(Input vector * Weight vector) + bias
	float probabilities = m_w[0]; // start the sum with the bias
	for(int i = 0; i < X.size(); i++)
		probabilities += X[i] * m_w[i+1];
	return probabilities;
}

// fit(vector< vector<float> > X, vector<float> y)
// Input: Matrix float X, vector float y
// Output: void
void perceptron::fit(std::vector< std:: vector<float> > X, vector<float> y)
{
	for(int i = 0; i < X[0].size() + 1; i++)
		m_w.push_back(0); // Setting each weight to 0
	for(int i = 0; i < m_epochs; i++)
		for(int j = 0; j < X.size(); j++)
		{
			float update = m_eta * (y[j] - predict(X[j])); // Calculate the change for the update
			for(int w = 1; w < m_w.size(); w++)
				m_w[w] += update * X[j][w-1];
			m_w[0] = update;
		}
}