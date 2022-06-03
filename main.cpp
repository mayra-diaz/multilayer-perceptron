#include "Perceptron.hpp"

int main(){
    auto mlp = Perceptron(2,int_vector{2,2});
    auto result = mlp.forward({2,3});
    for(auto val:result)
    {
        printf("%f", val);
    }

}