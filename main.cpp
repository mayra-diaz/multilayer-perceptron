#include "classes/Perceptron.hpp"

int main(){
    auto mlp = Perceptron(2,{2,2});
    auto result = mlp.forward({2,3});
    std::cout<<result<<'\n';
}