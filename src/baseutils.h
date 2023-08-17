// Basic utility functions for the project
#ifndef BASE_FUNCTIONS_INCLUDED
#define BASE_FUNCTIONS_INCLUDED

#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>


double gaussian(double x, double mu = 0.0, double sigma = 1.0);
double entropy(const std::vector<int>& x);

#pragma acc routine seq
double gaussian_gpu(double x, double mu, double sigma);

#endif

