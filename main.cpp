#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

// size1 = #filas y size2 = #columnas
int main(){
    auto df = parse_csv("butterflies55d.csv", 55);
    df.separation();
    auto mlp = Perceptron(55, {40, 30}, 10, 'r');
    mlp.train(df.X_train, df.Y_train, df.X_val, df.Y_val, 0.01, 100);
    mlp.test(df.X_test, df.Y_test);

}