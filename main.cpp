#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

// size1 = #filas y size2 = #columnas
int main(){
    auto df = parse_csv("butterflies96d.csv", 96);
    // std::cout<<df.X<<'\n';
    auto mlp = Perceptron({3, 2, 4, 5}, 'r');
    int n = 2, m = 3;
    mat input(n, m);
    auto result = mlp.forward(input);
    // cout << max(aux.data()) << endl;

    // cout << input.size1() << endl;
    // std::cout<<result<<'\n';
}