class perceptron
{
public:
	perceptron(float eta, int epochs);
	float netInput(std::vector<float> X);
	int predict(std::vector<float> X);
	// input is a matrix of inputs X and y is the feature we are looking for
	void fit(std::vector< std::vector<char> > X, std::vector<float> y);
private:
	float m_eta;
	int m_epochs;
	std::vector<float> m_weights; 
};