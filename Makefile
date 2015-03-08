# location of the Python header files
PYTHON_VERSION = 2.7
PYTHON_DIR = /Users/alexpark/anaconda
PYTHON_INCLUDE = $(PYTHON_DIR)/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE = $(PYTHON_DIR)/lib/python2.7/site-packages/numpy/core/include/numpy
# location of the Boost Python include files and library
BOOST_INC = /usr/local/include
BOOST_LIB = /usr/local/lib

NOBOOSTTARGET = test_noboost
$(NOBOOSTTARGET).so: $(NOBOOSTTARGET).o
	g++ -shared $(NOBOOSTTARGET).o -L$(PYTHON_DIR)/lib -lpython$(PYTHON_VERSION) -L/usr/local/lib -ljpeg -L/usr/local/cuda/lib `pkg-config opencv --libs` -o $(NOBOOSTTARGET).so

$(NOBOOSTTARGET).o: $(NOBOOSTTARGET).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I/usr/local/include -fPIC `pkg-config opencv --cflags-only-I` -c $(NOBOOSTTARGET).cpp


TARGET = nptest
#-Wl,--export-dynamic
$(TARGET).so: $(TARGET).o
	g++ -shared  $(TARGET).o -L$(BOOST_LIB) -lboost_python-$(PYTHON_VERSION) -L$(PYTHON_DIR)/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp
# 