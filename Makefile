all:
	pip install -e .
clean:
	pip uninstall pysplib
	rm -rdf build
	rm -f pysplib/*.so
	rm -drf *.egg-info
	rm -drf pysplib/__pycache__
test:
	python3 tests/test.py

