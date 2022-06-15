#include "../includes.hpp"

#ifndef CSVPARSER_H
#define CSVPARSER_H

struct DataFrame{
    mat X;
    vec Y;

    DataFrame(int num, int car);
};

DataFrame parse_csv(std::string filename, int dim);

#endif