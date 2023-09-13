#!/bin/bash -l

_g_input_file="interpolate.cu";
_g_output_file="interpolate_g$(python3-config --extension-suffix)";

_c_input_file="interpolate.cpp";
_c_output_file="interpolate_c$(python3-config --extension-suffix)";


[ -f ${_g_output_file} ] && rm ${_g_output_file}; 
[ -f ${_c_output_file} ] && rm ${_c_output_file};

pybind_include=$(python3 -m pybind11 --includes);


##############################################################################################################
##############################################################################################################
echo "Compiling the GPU version"
pgc++ -Wall -O2 -shared -std=c++17 -g -acc --diag_suppress set_but_not_used --diag_suppress declared_but_not_referenced --compiler-options ${pybind_include} ${_g_input_file} -o ${_g_output_file}

echo "Compiling the CPU version"
c++ -Wall -O3 -std=c++17 -g -fPIC -shared ${pybind_include} ${_c_input_file} baseutils.cpp -o ${_c_output_file}

if [ $? -eq 0 ] && [ -f ${_g_output_file} ] && [ -f ${_c_output_file} ]; then
  echo "Compilation successful";
  cp ${_g_output_file} /MieT5/BetaPose/nearl/static/;
  cp ${_c_output_file} /MieT5/BetaPose/nearl/static/;
  exit 0;
elif ! [ -f ${_g_output_file} ]; then 
  echo "GPU module compilation fails";
elif ! [ -f ${_c_output_file} ]; then 
  echo "CPU module compilation fails";
else
  echo "Error: Compilation failed";
  exit 1;
fi


#nvc++ -O3 -Xcompiler -Wall -shared -std=c++11 --compiler-options "-fPIC" ${pybind_include} ${input_file} -o ${output_file}
#nvcc -O3 -Xcompiler -Wall -shared -std=c++11 -g -gencode arch=compute_80,code=sm_80 --compiler-options "-fPIC" ${pybind_include} ${input_file} -o ${output_file}
#nvcc --verbose  --x cu --std=c++11 --shared  ${pybind_include} ${input_file} -o ${output_file}




