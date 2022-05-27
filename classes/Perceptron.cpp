#include"Perceptron.hpp"


// 4, [3, 2, 2]
// in:4 x nodes:3
// in:3 x nodes:2
// in:2 x nodes:2 -> output

Perceptron::Perceptron(int in_dim, vector<int> nodes)
{
    this->layers_n = nodes.size();
    this->in_dim = in_dim
    for (int i=0; i < n; ++i)
    {
        layers.push_back(new Layer(nodes[i], in_dim));
    }
}

void Perceptron::forward(){

}