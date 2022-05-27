#include "Layer.hpp"

Layer::Layer(int _node_dim, int _in_dim){
        node_dim = _node_dim;
        this->in_dim = _in_dim;
        srand(0);
        std::vector<double>* weight_row;
        for (int i = 0; i < node_dim; i++){
            weight_row = new std::vector<double>;
            for (int j = 0; j < in_dim; j++){
                weight_row->push_back((double)rand()/RAND_MAX);
            }
            weight_matrix->push_back(weight_row);
            bias_vec->push_back((double)rand()/RAND_MAX);
        }
    };