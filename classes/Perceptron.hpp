#include "../includes.hpp"

#ifndef PERCEPTRON_H
#define PERCEPTRON_H


class Perceptron{
    private:
        int n_layers;
        char activation_func_type = 'r';     // r = ReLU, s = Sigmoid, t = Tanh
        std::vector<mat> weights;
        std::vector<mat> weights_prime;
        std::vector<mat> deltas;
        std::vector<mat> layers_outputs;
        std::vector<vec> biases;


        mat activation_function(mat input, bool last)
        {
            // softmax
            if (last)
            {
                // cout << "softmax\n";
                FOR (i, 0, input.size1())
                {
                    double max = -1, sum = 1;
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
            else if (activation_func_type == 't')
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
        };
        mat activation_function_derivative(mat input)
        {
            // sigmoid
            if (activation_func_type == 's')
            {
                FOR (i, 0, input.size1())
                {
                    FOR (j, 0, input.size2())
                    {
                        // sigm(x) * (1 - sigm(x))
                        input(i, j) = (1/(1 + exp(-input(i, j)))) * (1 - (1/(1 + exp(-input(i, j)))));
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
                        // f(x) {x > 0 -> 1, x < 0 -> 0}
                        input(i, j) = input(i, j) > 0 ? 1 : 0;
                    }
                }
            } 
            // tanh
            else if (activation_func_type == 't')
            {
                FOR (i, 0, input.size1())
                {
                    FOR (j, 0, input.size2())
                    {
                        // 1-tanh(x)^2
                        input(i, j) = 1 - pow((exp(input(i, j)) - exp(-input(i, j)))/(exp(input(i, j)) + exp(-input(i, j))), 2);   
                    }                    
                }  
            }
            return input;
        };

        vec loss_derivative(vec input, bool last);

        vec derivates(int j);

    public:
        Perceptron(int in_dim, int_vector nodes, int out_dim, char _activation_func)
        {
            this->n_layers = nodes.size();
            this->activation_func_type = _activation_func;
            deltas.resize(n_layers+1);
            weights_prime.resize(n_layers+1);
            layers_outputs.resize(n_layers+1);
            weights.resize(n_layers+1);
            biases.resize(n_layers+1);

            // input layer
            biases[0] = (zero_vector<double>(nodes[0]));
            weights[0] = mat(in_dim, nodes[0]);
            FOR(j, 0, in_dim)
            {
                FOR(k, 0, nodes[0])
                {
                    weights[0](j, k) = ((double)rand()/RAND_MAX);
                }
            }

            // hidden layers
            FOR(i, 1, n_layers)
            {
                biases[i] = (zero_vector<double>(nodes[i]));
                weights[i] = mat(nodes[i-1], nodes[i]);
                FOR(j, 0, nodes[i-1])
                {
                    FOR(k, 0, nodes[i])
                    {
                        weights[i](j, k) = ((double)rand()/RAND_MAX);
                    }
                }
            }

            // output layer
            biases[n_layers] = (zero_vector<double>(out_dim));
            weights[n_layers] = mat(nodes[n_layers-1], out_dim);
            FOR(j, 0, nodes[n_layers-1])
            {
                FOR(k, 0, out_dim)
                {
                    weights[n_layers](j, k) = ((double)rand()/RAND_MAX);
                }
            }
            // cout << "finish\n";
        }

        mat forward(mat input)
        {
            FOR(i, 0, n_layers+1)
            {
                input = prod(input, weights[i]);
                FOR(j, 0, input.size1())
                {
                    FOR(k, 0, input.size2())
                    {
                        input(j, k) = input(j, k) + biases[i](k);
                    }
                }
                input = activation_function(input, i == n_layers);
                layers_outputs[i] = (input);
            }
            return input;
        }

        mat backward(mat input, b_vec_int y, double alpha)
        {
            deltas[n_layers] = layers_outputs[n_layers];
            weights_prime[n_layers] = (prod(trans(deltas[n_layers]), layers_outputs[n_layers-1]));

            ROF(i, n_layers-1, 0)
            {
                deltas[i] = element_prod(prod(deltas[i+1], trans(weights[i+1])), 
                                        activation_function_derivative(layers_outputs[i]));
                if (i != 0)
                {
                    weights_prime[i] = prod(trans(deltas[i]), layers_outputs[i-1]);
                }
                else
                {
                    weights_prime[i] = prod(trans(deltas[i]), input);
                }
            }
            FOR(i, 0, n_layers+1)
            {
                weights[i] = weights[i]-trans(weights_prime[i])*alpha;
                vec aux(deltas[i].size2());
                FOR(j, 0, deltas[i].size2())
                {
                    double sum = 0;
                    FOR(k, 0, deltas[i].size1())
                    {
                        sum += deltas[i](k, j);
                    }   
                    aux(j) = sum;
                }
                biases[i] = biases[i]-aux*alpha;
            }
            return input;
        }

        void train(mat x_train, b_vec_int y_train, mat x_val, b_vec_int y_val, double alpha, double epochs)
        {
            FOR(epoch, 1, epochs+1)
            {
                if(epoch%10 == 0)
                    cout << epoch << endl;
                forward(x_val);
                // cout << "----------------\n";
                forward(x_train);
                // cout << "----------------\n";
                backward(x_train, y_train, alpha);
            }
        }

        void test(mat x_test, b_vec_int y_test)
        {
            auto res = forward(x_test);
            double arg_max;
            vec result_vec(res.size1());
            FOR(i, 0, res.size1())
            {
                arg_max = -1;
                FOR(j, 0, res.size2())
                {
                    if(res(i,j) > arg_max)
                    {
                        arg_max = res(i,j);
                        result_vec(i) = j+1;
                    }
                }
            }
            cout<<result_vec<<'\n';
            cout<<y_test<<'\n';

            int score = 0;
            FOR(i, 0, result_vec.size()){
                if(result_vec(i) == y_test(i))
                    score++;
            }
            printf("Puntaje: %d/%lu correctos\n",score,result_vec.size());
        }
};

#endif 