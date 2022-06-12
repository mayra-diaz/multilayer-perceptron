#include "../includes.hpp"

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

class Perceptron{
    private:
        int layers_n = 0;
        int in_dim = 0;
        char activation_func_type = 'r';     // r = ReLU, s = Sigmoid, t = Tanh
        std::vector<mat*> layers{};
        std::vector<mat*> layers_linear{};
        std::vector<vec> layers_outputs{};
        std::vector<vec> biases{};


        vec activation_function(vec input, bool last);

        vec activation_function_derivative(vec input, bool last);

        vec loss_derivative(vec input, bool last);

        vec derivates(int j);

    public:
        Perceptron(int in_dim, int_vector nodes, char _activation_func);

        vec forward(vec input);

        vec backward(vec input, double alpha);
};

#endif 