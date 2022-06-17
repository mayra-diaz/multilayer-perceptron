#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

// size1 = #filas y size2 = #columnas
int main(){
    auto df = parse_csv("butterflies96d.csv", 96);
    df.separation();
    // std::cout<<df.Y_train<<'\n';
    auto mlp = Perceptron({3, 2, 4, 5}, 'r');
    // mlp.train(df.X_train, df.Y_train, df.X_val, df.Y_val, 0.01, 10);

}