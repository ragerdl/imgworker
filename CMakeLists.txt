cmake_minimum_required (VERSION 2.6)
project (ImgWorker)
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
find_package(Boost REQUIRED)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}/numpy
  ${PROJECT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
)

include_directories ("${PROJECT_SOURCE_DIR}")
add_library(ImgWorker imgworker.cpp)
