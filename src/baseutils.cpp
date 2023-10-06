#include <vector>
#include <omp.h>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <limits>
using namespace std;

double Gaussian(double x, double mu, double sigma) {
  const double sqrt_2_pi = sqrt(2.0 * M_PI);
  double a = (x - mu) / sigma;
  return exp(-0.5 * a * a) / (sigma * sqrt_2_pi);
}

double Entropy(const std::vector<int>& x){
	// Use a hashmap to count occurrences of each unique element
	if (x.size() == 1){return 0.0;}
	unordered_map<int, int> counts;
	for (const auto& xi : x){
		counts[xi]++;
	}
	// Calculate the probability of each unique element
	double total = static_cast<double>(x.size());
	vector<double> probs;
	for (const auto& pair : counts){
		probs.push_back(static_cast<double>(pair.second) / total);
	}
	// Calculate entropy using the formula: -sum(p * log2(p))
	double _entropy = 0.0;
	// Add a small constant to avoid log2(0)
	for (const auto& prob : probs){
		_entropy -= prob * log2(prob + 1e-10);
	}
	return _entropy;
}

template <typename T>
double CosineSimilarity(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if (vec1.size() != vec2.size()) {
		throw std::invalid_argument("Vectors must have the same length");
	}
	// Get the norm and the dot product of each vector
	const int length = vec1.size();
	double norm1 = 0.0;
	double norm2 = 0.0;
  double dot_product = 0.0;
	for (int i=0; i<length; ++i){
    dot_product += static_cast<double>(vec1[i] * vec2[i]);
		norm1 += static_cast<double>(vec1[i] * vec1[i]);
		norm2 += static_cast<double>(vec2[i] * vec2[i]);
	}
	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);

	if (norm1 * norm2 == 0){
		// zero division
		return 0.0;
	}
	return dot_product / (norm1 * norm2);
}


template <typename T>
bool almost_equal(const T& a, const T& b) {
	if constexpr (std::is_floating_point_v<T>) {
		const T epsilon = std::numeric_limits<T>::epsilon();
		return std::abs(a - b) < epsilon;
	} else {
		return a == b;
	}
}


template <typename T>
double JaccardSimilarity(const std::vector<T>& vec1, const std::vector<T>& vec2){
	if (vec1.empty() && vec2.empty()) return 1.0;
	if (vec1.empty() || vec2.empty()) return 0.0;

	// Sort the vectors because std::set_intersection requires sorted ranges
	std::vector<T> sorted_vec1(vec1);
	std::vector<T> sorted_vec2(vec2);
	std::sort(sorted_vec1.begin(), sorted_vec1.end());
	std::sort(sorted_vec2.begin(), sorted_vec2.end());

	std::vector<T> intersection;
	std::vector<T> union_vec;
	std::set_intersection(
			sorted_vec1.begin(), sorted_vec1.end(),
			sorted_vec2.begin(), sorted_vec2.end(),
			std::back_inserter(intersection));
	auto it1 = sorted_vec1.begin();
	auto it2 = sorted_vec2.begin();
	while (it1 != sorted_vec1.end() && it2 != sorted_vec2.end()) {
		if (almost_equal(*it1, *it2)) {
			intersection.push_back(*it1);
			union_vec.push_back(*it1);
			++it1;
			++it2;
		} else if (*it1 < *it2) {
			union_vec.push_back(*it1);
			++it1;
		} else {
			union_vec.push_back(*it2);
			++it2;
		}
	}
	// Append remaining elements to the union vector
	while (it1 != sorted_vec1.end()) {
		union_vec.push_back(*it1);
		++it1;
	}
	while (it2 != sorted_vec2.end()) {
		union_vec.push_back(*it2);
		++it2;
	}
	size_t union_size = union_vec.size();
	return static_cast<double>(intersection.size()) / static_cast<double>(union_size);
}


template <typename T>
double EuclideanSimilarity(const std::vector<T>& vec1, const std::vector<T>& vec2) {
	if (vec1.size() != vec2.size()) {
		throw std::runtime_error("Vectors must have the same length to compute Euclidean similarity");
	}
	double sum = 0.0;
	for (size_t i = 0; i < vec1.size(); ++i) {
		sum += (static_cast<double>(vec1[i]) - static_cast<double>(vec2[i])) * (static_cast<double>(vec1[i]) - static_cast<double>(vec2[i]));
	}
	return sqrt(sum);
}


template <typename T>
double ManhattanSimilarity(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Vectors must have the same length to compute Manhattan distance");
    }
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        sum += std::abs(static_cast<double>(vec1[i]) - static_cast<double>(vec2[i]));
    }
    return sum;
}


template <typename T>
std::vector<std::vector<double>> CosineSimilarityBatch(const std::vector<std::vector<T>>& vecs1, const std::vector<std::vector<T>>& vecs2){
	unsigned int len1 = vecs1.size();
	unsigned int len2 = vecs2.size();
	std::vector<std::vector<double>> similarities(len1, std::vector<double>(len2, 0.0));
	// Calculate the cosine similarity between each pair of vectors
	#pragma omp parallel for collapse(2)
	for (int i=0; i < vecs1.size(); ++i){
		for (int j=0; j < vecs2.size(); ++j){
			similarities[i][j] = CosineSimilarity<T>(vecs1[i], vecs2[j]);
		}
	}
	return similarities;
}

// NOTE: Instantiate the template for the its use in the pybind11 wrapper
template double CosineSimilarity<int>(const std::vector<int>& vec1, const std::vector<int>& vec2);
template double CosineSimilarity<float>(const std::vector<float>& vec1, const std::vector<float>& vec2);
template double CosineSimilarity<double>(const std::vector<double>& vec1, const std::vector<double>& vec2);
template std::vector<std::vector<double>> CosineSimilarityBatch<int>(const std::vector<std::vector<int>>& vecs1, const std::vector<std::vector<int>>& vecs2);
template std::vector<std::vector<double>> CosineSimilarityBatch<float>(const std::vector<std::vector<float>>& vecs1, const std::vector<std::vector<float>>& vecs2);
template std::vector<std::vector<double>> CosineSimilarityBatch<double>(const std::vector<std::vector<double>>& vecs1, const std::vector<std::vector<double>>& vecs2);

template <typename T>
std::vector<std::vector<double>> ManhattanSimilarityBatch(const std::vector<std::vector<T>>& vecs1, const std::vector<std::vector<T>>& vecs2) {
	unsigned int len1 = vecs1.size();
	unsigned int len2 = vecs2.size();
	std::vector<std::vector<double>> similarities(len1, std::vector<double>(len2, 0.0));
	// Calculate the cosine similarity between each pair of vectors
	#pragma omp parallel for collapse(2)
	for (int i=0; i < vecs1.size(); ++i){
		for (int j=0; j < vecs2.size(); ++j){
			similarities[i][j] = ManhattanSimilarity<T>(vecs1[i], vecs2[j]);
		}
	}
	return similarities;
}

template double ManhattanSimilarity<int>(const std::vector<int>& vec1, const std::vector<int>& vec2);
template double ManhattanSimilarity<float>(const std::vector<float>& vec1, const std::vector<float>& vec2);
template double ManhattanSimilarity<double>(const std::vector<double>& vec1, const std::vector<double>& vec2);
template std::vector<std::vector<double>> ManhattanSimilarityBatch<int>(const std::vector<std::vector<int>>& vecs1, const std::vector<std::vector<int>>& vecs2);
template std::vector<std::vector<double>> ManhattanSimilarityBatch<float>(const std::vector<std::vector<float>>& vecs1, const std::vector<std::vector<float>>& vecs2);
template std::vector<std::vector<double>> ManhattanSimilarityBatch<double>(const std::vector<std::vector<double>>& vecs1, const std::vector<std::vector<double>>& vecs2);

template <typename T>
std::vector<std::vector<double>> JaccardSimilarityBatch(const std::vector<std::vector<T>>& vecs1, const std::vector<std::vector<T>>& vecs2){
	unsigned int len1 = vecs1.size();
	unsigned int len2 = vecs2.size();
	std::vector<std::vector<double>> similarities(len1, std::vector<double>(len2, 0.0));
	// Calculate the cosine similarity between each pair of vectors
	#pragma omp parallel for collapse(2)
	for (int i=0; i < vecs1.size(); ++i){
		for (int j=0; j < vecs2.size(); ++j){
			similarities[i][j] = JaccardSimilarity<T>(vecs1[i], vecs2[j]);
		}
	}
	return similarities;
}

template double JaccardSimilarity<int>(const std::vector<int>& vec1, const std::vector<int>& vec2);
template double JaccardSimilarity<float>(const std::vector<float>& vec1, const std::vector<float>& vec2);
template double JaccardSimilarity<double>(const std::vector<double>& vec1, const std::vector<double>& vec2);
template std::vector<std::vector<double>> JaccardSimilarityBatch<int>(const std::vector<std::vector<int>>& vecs1, const std::vector<std::vector<int>>& vecs2);
template std::vector<std::vector<double>> JaccardSimilarityBatch<float>(const std::vector<std::vector<float>>& vecs1, const std::vector<std::vector<float>>& vecs2);
template std::vector<std::vector<double>> JaccardSimilarityBatch<double>(const std::vector<std::vector<double>>& vecs1, const std::vector<std::vector<double>>& vecs2);


template <typename T>
std::vector<std::vector<double>> EuclideanSimilarityBatch(const std::vector<std::vector<T>>& vecs1, const std::vector<std::vector<T>>& vecs2) {
	unsigned int len1 = vecs1.size();
	unsigned int len2 = vecs2.size();
	std::vector<std::vector<double>> distances(len1, std::vector<double>(len2, 0.0));
	// Calculate the cosine similarity between each pair of vectors
	#pragma omp parallel for collapse(2)
	for (int i=0; i < vecs1.size(); ++i){
		for (int j=0; j < vecs2.size(); ++j){
			distances[i][j] = EuclideanSimilarity<T>(vecs1[i], vecs2[j]);
		}
	}
	return distances;
}


template double EuclideanSimilarity<int>(const std::vector<int>& vec1, const std::vector<int>& vec2);
template double EuclideanSimilarity<float>(const std::vector<float>& vec1, const std::vector<float>& vec2);
template double EuclideanSimilarity<double>(const std::vector<double>& vec1, const std::vector<double>& vec2);
template std::vector<std::vector<double>> EuclideanSimilarityBatch<int>(const std::vector<std::vector<int>>& vecs1, const std::vector<std::vector<int>>& vecs2);
template std::vector<std::vector<double>> EuclideanSimilarityBatch<float>(const std::vector<std::vector<float>>& vecs1, const std::vector<std::vector<float>>& vecs2);
template std::vector<std::vector<double>> EuclideanSimilarityBatch<double>(const std::vector<std::vector<double>>& vecs1, const std::vector<std::vector<double>>& vecs2);


std::vector<int> SamplePoints(int N, int A, int B, bool sort_values = true){
	if (B - A + 1 < N) {
		throw std::invalid_argument("Range too small for the number of unique samples requested");
	}
	std::vector<int> values;
	for (int i = A; i <= B; ++i) {
		values.push_back(i);
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(values.begin(), values.end(), gen);
	values.resize(N);
	if (sort_values)
		std::sort(values.begin(), values.end());
	return values;
}



