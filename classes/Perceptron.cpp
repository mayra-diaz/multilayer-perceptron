#include"Perceptron.hpp"


// 4, [3, 2, 2]
// in:4 x nodes:3
// in:3 x nodes:2
// in:2 x nodes:2 -> output

Perceptron::Perceptron(int in_dim, int_vector nodes, char _activation_func)
{
    this->layers_n = nodes.size();
    this->in_dim = in_dim;
    this->activation_func_type = _activation_func;
    auto new_mat = new mat(nodes[0], in_dim+1);
    srand(time(NULL));
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

vec Perceptron::activation_function(vec input, bool last){
    vec output = input;
    if(last || activation_func_type == 's'){
        for(int i = 0; i < input.size(); i++) output[i] = 1/(1 + exp(-input[i]));
    }else if(activation_func_type == 'r'){
        for(int i = 0; i < input.size(); i++) output[i] = std::max((double) 0, input[i]);
    }else{
        for(int i = 0; i < input.size(); i++) output[i] = (exp(input[i]) - exp(-input[i]))/(exp(input[i]) + exp(-input[i]));   
    }
    return output;
}

vec Perceptron::forward(vec input){
    if(input.size() != this->in_dim){
        throw "Tamaño de input incorrecto.";
    }
    vec output = input;
    int last_counter = 1;
    for(auto layer:this->layers){
        auto size = output.size();
        output.resize(size+1);
        output.insert_element(size, 1);
        // std::cout<<"Input: "<<output<<'\n'<<"Matrix: "<<*layer<<'\n';
        output = activation_function(prod(*layer, output), this->layers.size() == last_counter);
        last_counter++;
    }
    return output;
}