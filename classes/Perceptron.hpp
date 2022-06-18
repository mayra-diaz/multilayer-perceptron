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
                // cout<<input<<'\n';
                // FOR (i, 0, input.size1())
                // {
                //     double max = -1, sum = 1;
                //     FOR (j, 0, input.size2())
                //     {
                //         if (input(i, j) > max) max = input(i, j);
                //         sum += input(i, j);
                //     }
                //     FOR (j, 0, input.size2())
                //     {
                //         input(i, j) = exp(input(i, j) - max) / sum;
                //     }
                // }
                double sum = 0;
                FOR(i, 0, input.size1())
                {
                    FOR(j, 0, input.size2())
                    {
                        sum += exp(input(i,j));
                    }
                    FOR(j, 0, input.size2())
                    {
                        input(i,j) = exp(input(i,j))/sum;
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
                zero_matrix<double> aux3(nodes[i], nodes[i+1]);
                weights_prime.push_back(aux3);
                deltas.push_back(aux3);
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
            FOR(i, 0, sz(weights))
            {
                // cout << input.size1() << "x" << input.size2() << " " << weights[i].size1() << "x" << weights[i].size2() << endl;
                input = prod(input, weights[i]);
                // cout << input.size2() << " " << biases[i].size() << endl;
                /* TODO: DA ERROR AL SUMAR EL BIAS, CORREGIR */ 
                // FOR(j, 0, input.size1())
                // {
                //     FOR(k, 0, input.size2())
                //     {
                //         input(j, k) = input(j, k) + biases[i](j);
                //     }
                // }
                input = activation_function(input, i == sz(weights)-1);
                layers_outputs.push_back(input);
            }
            return input;
        }

        mat backward(mat input, b_vec_int y, double alpha)
        {
            // deltas.back() = layers_outputs.back() - y;
            /* TODO: CONVERTIR y a matriz para poder restar */ 
            deltas.back() = layers_outputs.back();
            weights_prime.back() = trans(prod(trans(deltas.back()), layers_outputs[sz(layers_outputs)-2]));
            ROF(i, sz(weights)-1, 0)
            {
                deltas[i] = element_prod(prod(deltas[i+1], trans(weights[i+1])), 
                                         activation_function_derivative(layers_outputs[i]));
                weights_prime[i] = prod(trans(deltas[i]), input);
            }
            FOR(i, 0, sz(weights))
            {
                // cout << weights[i].size1() << "x" << weights[i].size2() <<
                //         " " << weights_prime[i].size1() << "x" << weights_prime[i].size2() << endl;
                weights[i] = weights[i]-(weights_prime[i])*alpha;
                /* TODO: FIXEAR ERROR CON DIMENSIONES */ 
                // vec aux(deltas[i].size2());
                // FOR(j, 0, deltas[i].size2())
                // {
                //     double sum = 0;
                //     FOR(k, 0, deltas[i].size1())
                //     {
                //         sum += deltas[i](k, j);
                //     }   
                //     aux(j) = sum;
                // }
                // biases[i] = biases[i]-aux*alpha;
            }
            return input;
        }

        void train(mat x_train, b_vec_int y_train, mat x_val, b_vec_int y_val, double alpha, double epochs)
        {
            FOR(epoch, 1, epochs+1)
            {
                if(epoch%10 == 0)
                    cout << epoch << endl;
                forward(x_train);
                forward(x_val);
                backward(x_train, y_train, alpha);
            }
        }

        void test(mat x_test, b_vec_int y_test)
        {
            auto res = forward(x_test);
            /* RESULTADO TODO -nan con tanh y relu, con sigmoid da resultados! */ 
            // cout << res << endl;
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