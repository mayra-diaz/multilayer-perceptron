#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

int main(){
    auto df = parse_csv("butterflies96d.csv", 96);
    // std::cout<<df.X<<'\n';
    auto mlp = Perceptron(2,{2,2}, 'r');
    auto result = mlp.forward({2,3});
    std::cout<<result<<'\n';
}