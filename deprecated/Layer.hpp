#include <vector>
#include <random>

class Layer{
private:
    std::vector<std::vector<double>*>* weight_matrix;
    std::vector<double>* bias_vec;
    int node_dim;
    int in_dim;

public:
    Layer(int _node_dim, int _in_dim);

    std::vector<double> activation_func(std::vector<double> inp, int i);



};