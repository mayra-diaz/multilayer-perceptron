#include "../includes.hpp"

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

template<typename T>
bool greatest_element(T a, T b)
{
    return a < b;
}

class Perceptron{
    private:
        int n_layers;
        char activation_func_type = 'r';     // r = ReLU, s = Sigmoid, t = Tanh
        std::vector<mat> weights;
        std::vector<mat> layers_outputs;
        std::vector<vec> biases;


        mat activation_function(mat input, bool last)
        {
            // softmax
            if (last)
            {
                FOR (i, 0, input.size1())
                {
                    double max = -1, sum = 0;
                    FOR (j, 0, input.size2())
                    {
                        if (input(i, j) > max) max = input(i, j);
                        sum += input(i, j);
                    }
                    FOR (j, 0, input.size2())
                    {
                        input(i, j) = exp(input(i, j) - max) / sum;
                    }
                }
            }
            // sigmoid
            else if (activation_func_type == 's')
            {
                FOR (i, 0, input.size1())
                {
                    FOR (j, 0, input.size2())
                    {
                        input(i, j) = 1/(1 + exp(-input(i, j)));
                    }
                }
            } 
            // relu
            else if (activation_func_type == 'r')
            {
                FOR (i, 0, input.size1())
                {
                    FOR (j, 0, input.size2())
                    {
                        input(i, j) = max((double) 0, input(i, j));
                    }
                }
            } 
            // tanh
            else 
            {
                FOR (i, 0, input.size1())
                {
                    FOR (j, 0, input.size2())
                    {
                        input(i, j) = (exp(input(i, j)) - exp(-input(i, j)))/(exp(input(i, j)) + exp(-input(i, j)));   
                    }                    
                }  
            }
            return input;
        }
        mat activation_function_derivative(mat input, int pos);

        vec loss_derivative(vec input, bool last);

        vec derivates(int j);

    public:
        Perceptron(int_vector nodes, char _activation_func)
        {
            this->n_layers = nodes.size();
            this->activation_func_type = _activation_func;
            
            FOR(i, 0, n_layers-1)
            {
                zero_vector<double> aux(nodes[i]);
                biases.push_back(aux);
                mat aux2(nodes[i], nodes[i+1]);
                FOR(j, 0, nodes[i])
                {
                    FOR(k, 0, nodes[i+1])
                    {
                        aux2(j, k) = ((double)rand()/RAND_MAX);
                    }
                }
                weights.push_back(aux2);
            }

            // DEBUG
            // FOR(i, 0, sz(weights))
            // {
            //     FOR(j, 0, nodes[i])
            //     {
            //         FOR(k, 0, nodes[i+1])
            //         {
            //             cout << weights[i](j, k) << " ";
            //         }
            //         cout << endl;
            //     }
            //     cout << endl;
            // }
            // cout << weights.size() << " " << biases.size() << endl;
        }

        mat forward(mat input)
        {
            FOR(i, 0, weights.size())
            {
                input = prod(input, weights[i]);
                // size1 = #filas y size2 = #columnas
                FOR(j, 0, weights[i].size1())
                {
                    input(i, j) = input(i, j) + biases[i](j);

                }
                input = activation_function(input, i == weights.size());
                layers_outputs.push_back(input);
            }
            return input;
        }

        mat backward(mat input, mat y)
        {
            // mat last = layers_outputs(n_layers) - y;
            return input;
        }
};

#endif 