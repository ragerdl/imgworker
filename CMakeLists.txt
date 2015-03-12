cmake_minimum_required (VERSION 2.6)
project (ImgWorker)

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release) # ensures -O3, no debugging symbols
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
find_package(JPEG REQUIRED)

set(Python_ADDITIONAL_VERSIONS "2.7")
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
             REQUIRED)
message("These are your python include dirs:\n${PYTHON_INCLUDE_DIRS}")

# no Numpy package in cmake, locate header dir manually
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import numpy as n; print(n.get_include());"
  RESULT_VARIABLE _NUMPY_SEARCH_SUCCESS
  OUTPUT_VARIABLE _NUMPY_VALUES
  ERROR_VARIABLE _NUMPY_ERROR_VALUE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_NUMPY_SEARCH_SUCCESS MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" _NUMPY_VALUES ${_NUMPY_VALUES})
  string(REGEX REPLACE "\n" ";" _NUMPY_VALUES ${_NUMPY_VALUES})
  list(GET _NUMPY_VALUES 0 NUMPY_INCLUDE_DIRS)
  string(REGEX REPLACE "\\\\" "/" NUMPY_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "NumPy import failure:\n${_NUMPY_ERROR_VALUE}")
endif()


set(BOOST_INCLUDEDIR "/usr/local/include")
set(BOOST_LIBRARYDIR "/usr/local/lib")
find_package(Boost REQUIRED COMPONENTS thread system)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}/numpy
  ${PROJECT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${JPEG_INCLUDE_DIR}
)


set(FINALIZE_DIR "/home/users/alex")
include_directories ("${PROJECT_SOURCE_DIR}")

add_library(ImgWorker SHARED imgworker.cpp)
set_target_properties(ImgWorker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set_target_properties(ImgWorker PROPERTIES PREFIX "_")
target_link_libraries(ImgWorker
  ${PYTHON_LIBRARIES}
  ${JPEG_LIBRARIES}
  ${Boost_LIBRARIES}
)
message("SDIR:  sourcedir: ${PROJECT_SOURCE_DIR} builddir: ${PROJECT_BINARY_DIR}")
install(TARGETS ImgWorker LIBRARY DESTINATION ${FINALIZE_DIR})
