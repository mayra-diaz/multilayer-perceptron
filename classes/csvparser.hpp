#include "../includes.hpp"

#ifndef CSVPARSER_H
#define CSVPARSER_H

struct DataFrame{
    mat X;
    vec Y;

    mat X_train;
    mat X_val;
    mat X_test;

    vec Y_train;
    vec Y_val;
    vec Y_test;

    DataFrame(int num, int car);
    
    // TODO: que la separación sea RANDOM
    void separation()
    {
        int train = Y.size()*0.8;
        FOR(i, 0, train)
        {
            Y_train(i) = Y(i);
            X_train(i) = X(i);
        }
        int val = Y.size()*0.1;
        FOR(i, train, train+val)
        {
            Y_val(i-train) = Y(i);
            X_val(i-train) = X(i-train);
        }
        int test = Y.size()*0.1;
        FOR(i, train+val, train+val+test)
        {
            Y_test(i-train-val) = Y(i);
            X_test(i-train-val) = X(i-train-val);
        }
    }
};

DataFrame parse_csv(std::string filename, int dim);

#endif