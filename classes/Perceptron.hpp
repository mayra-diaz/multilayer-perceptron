#include "../includes.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include"Layer.hpp"

class Perceptron{
    private:
        int layers_n = 0;
        int in_dim = 0;
        std::vector<mat*> layers;

    public:
        Perceptron(int in_dim, int_vector nodes);

        vec forward(vec input);
};