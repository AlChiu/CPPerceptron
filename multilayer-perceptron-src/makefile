mlp: filereader.o mlp.o main.o
	g++ filereader.o mlp.o main.o -o mlp

filereader.o: filereader.cpp filereader.h
	g++ -c filereader.cpp

mlp.o: mlp.cpp mlp.h
	g++ -c mlp.cpp

main.o: main.cpp
	g++ -c main.cpp