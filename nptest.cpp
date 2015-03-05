#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <arrayobject.h>
#include <stdexcept>

namespace bp = boost::python;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

int divideByTwoSum(bp::list *pylist)
{
    int n = bp::extract<int>(pylist->attr("__len__")());
    int dsum = 0;
    for ( int i = 0; i < n; i++ ){
        dsum += (bp::extract<int>((*pylist)[i]));
    }
	return dsum;
}

BOOST_PYTHON_MODULE(nptest)
{
	bp::def("divideByTwoSum", divideByTwoSum);
}