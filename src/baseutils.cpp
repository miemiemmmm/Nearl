#include "baseutils.h"
using namespace std;

double gaussian(double x, double mu, double sigma) {
  const double sqrt_2_pi = sqrt(2.0 * M_PI);
  double a = (x - mu) / sigma;
  return exp(-0.5 * a * a) / (sigma * sqrt_2_pi);
}


double entropy(const std::vector<int>& x){
	// Use a hashmap to count occurrences of each unique element
	if (x.size() == 1){return 0.0;}
	unordered_map<int, int> counts;
	for (const auto& xi : x){counts[xi]++; }
	// Calculate the probability of each unique element
	double total = static_cast<double>(x.size());
	vector<double> probs;
	for (const auto& pair : counts){probs.push_back(static_cast<double>(pair.second) / total);}
	// Calculate entropy using the formula: -sum(p * log2(p))
	double _entropy = 0.0;
	// Add a small constant to avoid log2(0)
	for (const auto& prob : probs){_entropy -= prob * log2(prob + 1e-10);}
	return _entropy;
}

