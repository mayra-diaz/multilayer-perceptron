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
    auto new_mat = new mat(nodes[0], in_dim);
    srand(time(NULL));

    biases = std::vector<vec> (nodes.size());
    for (int i = 0; i < biases.size(); i++)
        biases[i] = vec(nodes[i]);

    for (int i = 0; i < nodes[0]; i++)
    {
        for (int j = 0; j < in_dim; j++){
            (*new_mat)(i,j) = (double)rand()/RAND_MAX;
        }
    }
    layers.push_back(new_mat);
    for (int i = 1; i < layers_n; ++i){
        new_mat = new mat(nodes[i], nodes[i-1]);
        for (int i = 0; i < nodes[i]; i++){
            for (int j = 0; j < nodes[i-1]; j++){
               (*new_mat)(i,j) = (double)rand()/RAND_MAX;
            }
        }
        layers.push_back(new_mat);
    }
}

vec Perceptron::activation_function(vec input, bool last=false){
    vec output(input.size());
    if(last || activation_func_type == 's'){
        for(int i = 0; i < input.size(); i++) output[i] = 1/(1 + exp(-input[i]));
    }else if(activation_func_type == 'r'){
        for(int i = 0; i < input.size(); i++) output[i] = std::max((double) 0, input[i]);
    }else{
        for(int i = 0; i < input.size(); i++) output[i] = (exp(input[i]) - exp(-input[i]))/(exp(input[i]) + exp(-input[i]));   
    }
    return output;
}

int Perceptron::activation_function_derivative(vec input, bool last=false){ 
    int output = 0;
    if(last || activation_func_type == 's'){
        for(int i = 0; i < input.size(); i++) output += input[i]*(1-input[i]);
    }else if(activation_func_type == 'r'){
        for(int i = 0; i < input.size(); i++) output +=  input[i] > 0 ? 1 : 0;
    }else{
        for(int i = 0; i < input.size(); i++) output += 1-sqrt(input[i]);   
    }
    return output;
}


vec Perceptron::derivates(int j, bool last=false){
    vec output(*(layers_linear[j]).size());
    for(int i = 0; i < *(layers_linear[j]).size(); i++) output[i] = activation_function_derivative(layers_linear[i], last);
    return prod(trans(*(layers[j+1])), biases[j+1]) * output;
}

vec Perceptron::forward(vec input)
{
    if (input.size() != this->in_dim)
    {
        throw "Tamaño de input incorrecto.";
    }
    vec output(input.size());
    int last_counter = 1;
    for (int i = 0; i < layers.size(); i++)    // TODO: poner el last abajo
    {
        // std::cout<<"Input: "<<output<<'\n'<<"Matrix: "<<*layers[i]<<'\n';
        vec dot_product = prod(*(layers[i]), output)+biases[i];
        layers_linear.push_back(dot_product);
        output = activation_function(dot_product, this->layers.size() == last_counter);
        last_counter++;
        layers_outputs.push_back(output);
    }
    return output;
}

vec Perceptron::backward(vec input, vec y, double alpha)
{
    std::vector<double> derivatives;
    vec<mat*> w = new mat(*(layers[0]), layers.size());
    
    biases[-1] = (layers_outputs[-1]-y)*activation_function_derivative(layers_linear[-1], true);
    *(w[-1]) = trans(prod(biases[-1], layers_output[-2]));
    for (int i = layers.size()-2; i >= 0; --i)
    {
        biases[i] = derivatives(i);
        *(w[i]) = prod(biases[i], trans(layers_outputs[i - 1]));
    }
}
