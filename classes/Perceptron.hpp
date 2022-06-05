#include "../includes.hpp"

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

class Perceptron{
    private:
        int layers_n = 0;
        int in_dim = 0;
        char activation_func_type = 'r';     // r = ReLU, s = Sigmoid, t = Tanh
        std::vector<mat*> layers;

    public:
        Perceptron(int in_dim, int_vector nodes, char _activation_func);

        vec activation_function(vec input, bool last);

        vec forward(vec input);
};

#endif 