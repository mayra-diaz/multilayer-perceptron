#include "csvparser.hpp"

DataFrame::DataFrame(int num, int car){
    this->X = mat(num,car);
    this->Y = vec(num);
}

DataFrame parse_csv(std::string filename, int dim){
    std::ifstream file(filename);
    if(file.fail()) throw "Error al abrir el archivo";
    std::string line;
    int num = 0;
    while(std::getline(file, line)) num++;
    file.close();
    file.open(filename);
    DataFrame df(num, dim);
    int line_counter = 0, value_counter;
    while(std::getline(file, line)){
        size_t pos_start = 0, pos_end;
        value_counter = 0;
        while ((pos_end = line.find (',', pos_start)) != std::string::npos) {
            df.X(line_counter, value_counter) = std::stod(line.substr (pos_start, pos_end - pos_start));
            pos_start = pos_end + 1;
            value_counter += 1;
        }
        df.Y(line_counter) = std::stod(line.substr(pos_start));
        line_counter += 1;
    }
    return df;
}