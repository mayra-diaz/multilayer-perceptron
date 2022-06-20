#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <boost/assign/std/vector.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/foreach.hpp>

using namespace boost::assign;
using namespace boost::numeric::ublas;
using namespace std;

# define int_vector std::vector<int>
# define mat boost::numeric::ublas::matrix<double>
# define vec boost::numeric::ublas::vector<double>
# define b_vec_int boost::numeric::ublas::vector<int>
# define db double
#define FOR(i,a,b) for (int (i) = (a); (i) < (b); ++(i))
#define ROF(i,a,b) for (int (i) = (a); (i) >= (b); --(i))
#define sz(x) int((x).size())
// using std::vector<int> = int_vector;
// using boost::numeric::ublas::matrix<double> = mat;
// using boost::numeric::ublas::vector<double> = vec;