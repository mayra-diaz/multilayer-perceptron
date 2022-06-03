#include"Perceptron.hpp"


// 4, [3, 2, 2]
// in:4 x nodes:3
// in:3 x nodes:2
// in:2 x nodes:2 -> output

Perceptron::Perceptron(int in_dim, int_vector nodes)
{
    this->layers_n = nodes.size();
    this->in_dim = in_dim;
    auto new_mat = new matrix<double>(nodes[0], in_dim+1)
    for (int i = 0; i < nodes[0]; i++){
        for (int j = 0; j < in_dim+1; j++){
            (*new_mat)(i,j) = (double)rand()/RAND_MAX;
        }
    }
    layers.push_back(new_mat);
    for (int i = 1; i < n; ++i){
        new_mat = new matrix<double>(nodes[i], nodes[i-1]+1);
        for (int i = 0; i < nodes[i]; i++){
            for (int j = 0; j < nodes[i-1]+1; j++){
               (*new_mat)(i,j) = (double)rand()/RAND_MAX;
            }
        }
        layers.push_back(new_mat);
    }
}

vec Perceptron::forward(vec input){
    if(input.size() != this->in_dim){
        printf("Tamaño de input incorrecto.")
        return;
    }
    vec *output;
    *output = input;
    for(auto layer:this->layers){
        *output += 1;
        *output = prod(layer, *output);
        
         BOOST_FOREACH( auto v, *output ){
            std::cout << v << ' ';
        }
        cout << '\n';
    }
    return *output;
}