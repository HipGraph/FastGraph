rm -f *.so
g++ -O3 -Wall -std=c++17 -shared -fPIC -fopenmp -g $(python3 -m pybind11 --includes) COO_pybind.cpp -o COO$(python3-config --extension-suffix) $(python3-config --ldflags)
g++ -O3 -Wall -std=c++17 -shared -fPIC -fopenmp -g $(python3 -m pybind11 --includes) CSC_pybind.cpp -o CSC$(python3-config --extension-suffix) $(python3-config --ldflags)
g++ -O3 -Wall -std=c++17 -shared -fPIC -fopenmp -g $(python3 -m pybind11 --includes) CSR_pybind.cpp -o CSR$(python3-config --extension-suffix) $(python3-config --ldflags)