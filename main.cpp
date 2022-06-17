#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

int main(){
    auto df = parse_csv("butterflies96d.csv", 96);
    // std::cout<<df.X<<'\n';
    auto mlp = Perceptron({3, 2, 4}, 'r');
    int n = 2, m = 2;
    // mat input(n, m);
    // cout << max(aux.data()) << endl;

    // cout << input.size1() << endl;
    // auto result = mlp.forward({2,3});
    // std::cout<<result<<'\n';
}