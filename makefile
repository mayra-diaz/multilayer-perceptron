make:
	g++ -o test.exe main.cpp classes/Perceptron.cpp classes/csvparser.cpp -O3
	./test.exe
	rm test.exe

clear:
	clear
	g++ -o test.exe main.cpp classes/Perceptron.cpp classes/csvparser.cpp -O3
	./test.exe
	rm test.exe