#include<vector>
#include <random>
#include <boost/assign/std/vector.hpp>
#include <boost/foreach.hpp>

using namespace boost::assign;

using namespace std;

using std::vector<int> = int_vector;
using boost::numeric::ublas::matrix<double> = mat;
using boost::numeric::ublas::vector<double> = vec;

int suma(int a, int b) {
    return a + b;
}

auto resta = [](int a, int b){
    return a - b;
};