#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>


using namespace std;
namespace py = pybind11;

void template_func(){
	// do something
	std::cout << "This is the template function" << std::endl;
}

PYBIND11_MODULE(template, m) {  // <- replace module name
  m.def("template", &template_func,
  "template function");   // <- replace function information
}


/*
Template command to compile the code into a shared library
g++ -std=c++17 -O3 -shared -fPIC -fopenmp -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include pybind_template.cpp -o template$(python3-config --extension-suffix)
*/