// Created by: Yang Zhang
// Description: Basic CPU utility functions for the project
#ifndef CPU_UTILS_INCLUDED
#define CPU_UTILS_INCLUDED

#include <omp.h>
#include <cmath>

#include <vector>
#include <random>
#include <limits>

#include <iostream>
#include <algorithm>
#include <unordered_map>


template <typename T>
T gaussian_map(T dist, double mu = 0.0, double sigma = 1.0){
  // Distance-based gaussian probability of x given mu and sigma
  return std::exp(-0.5 * std::pow((dist - mu) / sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
}


template <typename T>
T max(const T *Arr, const int N){
  T max_val = Arr[0];
  for (int i = 1; i < N; ++i){
    if (Arr[i] > max_val) {
      max_val = Arr[i];
    }
  }
  return max_val;
}


template <typename T>
T min(const T *Arr, const int N){
  T min_val = Arr[0];
  for (int i = 1; i < N; ++i) {
    if (Arr[i] < min_val) {
      min_val = Arr[i];
    }
  }
  return min_val;
}


template <typename T>
T sum(const T *Arr, const int N){
  T sum_val = 0;
  for (int i = 0; i < N; ++i) {
    sum_val += Arr[i];
  }
  return sum_val;
}


template <typename T>
T mean(const T *Arr, const int N){
  T thesum = sum(Arr, N);
  return thesum / N;
}


template <typename T>
T standard_deviation(const T *x, const int n){
  T mean_val = mean(x, n);
  T sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += (x[i] - mean_val) * (x[i] - mean_val);
  }
  return std::sqrt(sum / n);
}


template <typename T>
T variance(const T *x, const int n){
  T _mean = mean(x, n);
  T sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += (x[i] - _mean) * (x[i] - _mean);
  }
  return sum / n;
}


template <typename T>
T median(const T *x, const int n) {
  T *y = new T[n];
  for (int i = 0; i < n; ++i) {
    y[i] = x[i];
  }
  std::sort(y, y + n);
  T median;
  if (n % 2 == 0) {
    median = (y[n / 2 - 1] + y[n / 2]) / 2;
  } else {
    median = y[n / 2];
  }
  delete[] y;
  return median;
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


template <typename T>
bool almost_equal(const T& a, const T& b) {
  if constexpr (std::is_floating_point_v<T>) {
    const T epsilon = std::numeric_limits<T>::epsilon();
    return std::abs(a - b) < epsilon;
  } else {
    return a == b;
  }
}


/**
 * @brief Get the information entropy of a vector of integers
 */ 
inline double information_entropy(const std::vector<int>& Arr){
  if (Arr.size() <= 1){ return 0.0; }
  std::unordered_map<int, int> counts;
  for (const auto& xi : Arr){
    counts[xi]++;
  }
  // Calculate the probability of each unique element
  double entropy_val = 0.0;
  for (const auto& pair : counts){
    double prob = static_cast<double>(pair.second) / Arr.size();
    entropy_val -= prob * log2(prob + 1e-10);
  }
  return entropy_val;
}

/**
 * This is the same information entropy realization as the information_entropy_device 
 * in the gpuutils.cuh
 */
template <typename T>
double information_entropy(const T *Arr, const int N){
  if (N <= 1){ return 0.0f; }
  int *Arr_copy = new int[N];
  for (int i = 0; i < N; i++){
    Arr_copy[i] = static_cast<int>(Arr[i]*10);
  }
  std::unordered_map<int, int> counts;
  for (int i = 0; i < N; i++){
    counts[Arr_copy[i]]++;
  }
  double entropy_val = 0.0;
  for (const auto& pair : counts){
    double prob = static_cast<double>(pair.second) / N;
    entropy_val -= prob * log2(prob + 1e-10);
  }
  delete[] Arr_copy;
  return entropy_val;
}


// Used in ICP algorithm
inline std::vector<int> sample_points(int N, int A, int B, bool sort_values = true){
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
  if (sort_values){
    std::sort(values.begin(), values.end());
  }
  return values;
}


inline void translate_coord(float* coord, const int atom_nr, const int *dims, const double spacing) {
  // Align the center of the coords to the center of the grid
  float cog_coord[3] = {0};
  for (int i = 0; i < atom_nr; i++) {
    cog_coord[0] += coord[i*3];
    cog_coord[1] += coord[i*3+1];
    cog_coord[2] += coord[i*3+2];
  }
  cog_coord[0] /= atom_nr;
  cog_coord[1] /= atom_nr;
  cog_coord[2] /= atom_nr;

  for (int i = 0; i < atom_nr; i++) {
    coord[i*3]   = (coord[i*3] - cog_coord[0]) + (dims[0]*spacing)/2;
    coord[i*3+1] = (coord[i*3+1] - cog_coord[1]) + (dims[1]*spacing)/2;
    coord[i*3+2] = (coord[i*3+2] - cog_coord[2]) + (dims[2]*spacing)/2;
  }
}


#endif

