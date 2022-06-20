#include "../includes.hpp"

#ifndef CSVPARSER_H
#define CSVPARSER_H

struct DataFrame{
    mat X;
    b_vec_int Y;

    mat X_train;
    mat X_val;
    mat X_test;

    b_vec_int Y_train;
    b_vec_int Y_val;
    b_vec_int Y_test;

    DataFrame(int num, int car);
    
    // TODO: que la separación sea RANDOM
    void separation()
    {
        srand(time(NULL));
        b_vec_int index(Y.size());
        FOR(i, 0, Y.size()) index(i) = i;
        random_shuffle(index.begin(), index.end());
        int train = Y.size()*0.8;
        FOR(i, 0, train)
        {
            Y_train(i) = Y(index(i));
            FOR(j, 0, X.size2())
                X_train(i,j) = X(index(i),j);
        }
        int val = Y.size()*0.1;
        FOR(i, train, train+val)
        {
            Y_val(i-train) = Y(index(i));
            FOR(j, 0, X.size2())
                X_val(i-train,j) = X(index(i),j);
        }
        int test = Y.size()*0.1;
        FOR(i, train+val, train+val+test)
        {
            Y_test(i-train-val) = Y(index(i));
            FOR(j, 0, X.size2())
                X_test(i-train-val,j) = X(index(i),j);
        }
        // cout<<X_test<<'\n';
        // exit(0);
    }
};

DataFrame parse_csv(std::string filename, int dim);

#endif