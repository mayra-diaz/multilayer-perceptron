#include "classes/Perceptron.hpp"
#include "classes/csvparser.hpp"

// size1 = #filas y size2 = #columnas
int main(){
    auto df = parse_csv("butterflies96d.csv", 96);
    df.separation();
    auto mlp = Perceptron({96, 60, 40, 10}, 's');
    mlp.train(df.X_train, df.Y_train, df.X_val, df.Y_val, 0.01, 10);
    mlp.test(df.X_test, df.Y_test);

}