all:
	sudo pip install -e .
clean:
	sudo pip uninstall mysplib
	sudo rm -rdf build
	sudo rm mysplib/*.so
test:
	sudo python3 tests/test.py

