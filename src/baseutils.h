// Basic utility functions for the project
#ifndef BASE_FUNCTIONS_INCLUDED
#define BASE_FUNCTIONS_INCLUDED

// Get the gaussian probability of x given mu and sigma
double Gaussian(double x, double mu = 0.0, double sigma = 1.0);

// Get the entropy of a vector of integers (int !!!!)
double Entropy(const std::vector<int>& x);


template <typename T>
double CosineSimilarity(const std::vector<T>& vec1, const std::vector<T>& vec2);

template <typename T>
std::vector<std::vector<double>> CosineSimilarityBatch(const std::vector<std::vector<T>>& vecs1, const std::vector<std::vector<T>>& vecs2);

std::vector<int> SamplePoints(int N, int A, int B, bool sort_values = true);

#endif

