#include"Perceptron.hpp"


// 4, [3, 2, 2]
// in:4 x nodes:3
// in:3 x nodes:2
// in:2 x nodes:2 -> output

Perceptron::Perceptron(int in_dim, int_vector nodes)
{
    this->layers_n = nodes.size();
    this->in_dim = in_dim;
    auto new_mat = new mat(nodes[0], in_dim+1);
    for (int i = 0; i < nodes[0]; i++){
        for (int j = 0; j < in_dim+1; j++){
            (*new_mat)(i,j) = (double)rand()/RAND_MAX;
        }
    }
    layers.push_back(new_mat);
    for (int i = 1; i < layers_n; ++i){
        new_mat = new mat(nodes[i], nodes[i-1]+1);
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
        throw "Tamaño de input incorrecto.";
    }
    vec output = input;
    for(auto layer:this->layers){
        auto size = output.size();
        output.resize(size+1);
        output.insert_element(size, 1);
        // std::cout<<output<<'\n'<<*layer<<'\n';
        output = prod(*layer, output);
    }
    return output;
}