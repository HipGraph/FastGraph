# Welcome to The documentation of SpLib

SpLib is a A parallel Sparse Matrix Library. This is being developed by HipGraph lab. For contact information visit [HipGraph Lab](https://hipgraph.luddy.indiana.edu/).

## Structure

The library is very flexible and interested users can develop custom function both in C++ and Python and utilize the core functionalities of SpLib. You can develop C++ codes in /csrc, and Python codes in /pysplib.

## Requirements

- `C++ 11` or above
- `Pybind11`
- `Python 3.7` or above

`sudo` permission is also required. 

## Installation and Testing

- Run `git clone` to clone the repository
- Switch to `python_wrapper` branch
- Run `make` to install
- Run `make test` to run test file

## Uninstall 

- Run `make clean`

## Project layout

    csrc/	  		# C++ functions and data structures
        .	  		# C++ operations built on top of core functions of SpLib
        common/   		# Core SpLib C++ files and data structures such as CSR and CSC.
	pybind_wrapper.cpp 	# Select which functions to expose in Python package

    pysplib/	  		# Python wrapper for SpLib
	init.py   		# Initializers
    tests			# Contains test cases
    setup.py  			# Run this file to 
    docs/			# Project documentations



## COO:
    Coordinate Format Sparse Matrix Class. 

## CSR

    1. CSR(COO<RIT, CIT, VT> & cooMat);
       Return 
