make:
	g++ -o test.exe main.cpp classes/Perceptron.cpp classes/csvparser.cpp
	./test.exe
	rm test.exe