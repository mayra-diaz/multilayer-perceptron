#include "../includes.hpp"
#include"Layer.hpp"

class Perceptron{
    private:
        int layers_n = 0;
        int in_dim = 0;
        vector<Layer *> layers{};

    public:
        Perceptron(vector<int> nodes);

        void forward();
};