# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.5)
project (pyscf)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
endif()

option(BUILD_WITH_AVX2 "set avx2 flags" OFF)
if(BUILD_WITH_AVX2)
  include(CheckCCompilerFlag)
  if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    CHECK_C_COMPILER_FLAG("-axAVX2" COMPILER_SUPPORTS_AVX)
    if (COMPILER_SUPPORTS_AVX)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -axAVX2")
    endif()
  else()
    CHECK_C_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX)
    if (COMPILER_SUPPORTS_AVX)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -ftree-vectorize")
    endif()
  endif()
endif()

option(BUILD_WITH_AVX512 "set avx512 flags" OFF)
if(BUILD_WITH_AVX512)
  include(CheckCCompilerFlag)
  if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    CHECK_C_COMPILER_FLAG("-axCORE-AVX512" COMPILER_SUPPORTS_AVX)
    if (COMPILER_SUPPORTS_AVX)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -axCORE-AVX512")
    endif()
  else()
    CHECK_C_COMPILER_FLAG("-mavx512f -mavx512dq" COMPILER_SUPPORTS_AVX)
    if (COMPILER_SUPPORTS_AVX)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512dq")
    endif()
  endif()
endif()


set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Architecture specified cmake flags.  See also the templates in
# pyscf/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

if (NOT BLAS_LIBRARIES)
#enable_language(Fortran)
find_package(BLAS)
check_function_exists(ffsll HAVE_FFS)
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()
# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "                  /path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_def.so")
# set(BLAS_LIBRARIES "-Wl,-rpath=${MKLROOT}/lib/intel64/ ${BLAS_LIBRARIES}")
#
# These settings can be written in the cmake.arch.inc file. This config file
# will automatically load all settings specified in cmake.arch.inc
#

find_package(OpenMP)
if(OPENMP_FOUND)
  set(HAVE_OPENMP 1)
else ()
  set(OpenMP_C_FLAGS " ")
endif()

#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
#  OUTPUT_VARIABLE NUMPY_INCLUDE)
#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

option(ENABLE_FFTW "Using fftw3" ON)
option(BUILD_FFTW "Building fftw3" ON)

add_subdirectory(np_helper)
add_subdirectory(gto)
add_subdirectory(vhf)
add_subdirectory(ao2mo)
add_subdirectory(mcscf)
add_subdirectory(cc)
add_subdirectory(ri)
#add_subdirectory(localizer)
add_subdirectory(pbc)
add_subdirectory(agf2)

# Overwrite CMAKE_C_CREATE_SHARED_LIBRARY in Modules/CMakeCInformation.cmake
# to remove the SONAME flag in the so file. The soname information causes
# dynamic link error when importing libcint library.
set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

include(ExternalProject)
option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" ON)
if(BUILD_LIBCINT)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/cint.h")
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/fishjojo/libcint.git
    GIT_TAG dev
    # GIT_REPOSITORY https://github.com/sunqm/libcint.git # libcint is a portable, cross-platform implementation
    # GIT_REPOSITORY https://github.com/sunqm/qcint.git # qcint is an optimized implementation for x86-64 architecture
    # GIT_TAG v4.4.6
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_CACHE_ARGS 
        -DWITH_F12:STRING=${WITH_F12} 
        -DWITH_RANGE_COULOMB:STRING=1
        -DWITH_FORTRAN:STRING=0 
        -DWITH_CINT2_INTERFACE:STRING=0 
        -DMIN_EXPCUTOFF:STRING=20
        -DKEEP_GOING:STRING=1
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR:PATH=lib 
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DBLAS_LIBRARIES:STRING=${BLAS_LIBRARIES}
        -DCMAKE_C_CREATE_SHARED_LIBRARY:STRING=${C_LINK_TEMPLATE}
        -DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}
  )
  add_dependencies(cgto libcint)
  add_dependencies(ao2mo libcint)
endif()
endif()

option(ENABLE_LIBXC "Using libxc for XC functional library" ON)
option(ENABLE_XCFUN "Using xcfun for XC functional library" ON)
option(BUILD_LIBXC "Download and build libxc library" ON)
option(BUILD_XCFUN "Download and build xcfun library" ON)

option(ENABLE_LIBXSMM "Using libxsmm" OFF)
option(BUILD_LIBXSMM "Building libxsmm" ON)
if(APPLE)
    set(ENABLE_LIBXSMM OFF)
endif()

if(NOT DISABLE_DFT)
add_subdirectory(dft)

#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xc.h" AND
#    ENABLE_LIBXC AND BUILD_LIBXC)
if(ENABLE_LIBXC AND BUILD_LIBXC)
  ExternalProject_Add(libxc
    #GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
    #GIT_TAG master
    URL https://gitlab.com/libxc/libxc/-/archive/5.1.7/libxc-5.1.7.tar.gz
    #URL https://github.com/MoleOrbitalHybridAnalyst/libxc_copy/archive/refs/tags/5.1.5m2.tar.gz
    #URL https://github.com/MoleOrbitalHybridAnalyst/libxc_copy/archive/refs/heads/main.zip
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DENABLE_FORTRAN=0 -DDISABLE_KXC=0 -DDISABLE_LXC=1
            -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
  )
  add_dependencies(xc_itrf libxc)
  add_dependencies(dft libxc)
endif() # ENABLE_LIBXC

#if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/xcfun.h" AND
#    ENABLE_XCFUN AND BUILD_XCFUN)
if(ENABLE_XCFUN AND BUILD_XCFUN)
  ExternalProject_Add(libxcfun
    #GIT_REPOSITORY https://github.com/sunqm/xcfun.git
    GIT_REPOSITORY https://github.com/fishjojo/xcfun.git
    # copy of v2.1.1, downgrade to cmake 3.5
    GIT_TAG cmake-3.5
    # Range seperated parameters can be set in the python code. This patch to
    # RSH omega is not needed anymore.
    #PATCH_COMMAND patch -p1 < ${PROJECT_SOURCE_DIR}/libxcfun.patch
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
            -DXCFUN_MAX_ORDER=3 -DENABLE_TESTALL=0
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_CXX_CREATE_SHARED_LIBRARY=${CXX_LINK_TEMPLATE}
  )
  add_dependencies(xcfun_itrf libxcfun)
  add_dependencies(dft libxcfun)
endif() # ENABLE_XCFUN

if(ENABLE_LIBXSMM AND BUILD_LIBXSMM)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/libxsmm.h")
    ExternalProject_Add(libxsmm
      GIT_REPOSITORY https://github.com/hfp/libxsmm.git
      GIT_TAG 1.17
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE True
      BUILD_COMMAND make -j4 PREFIX=<INSTALL_DIR> CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} STATIC=0 MALLOC=0 INTRINSICS=2 install
      INSTALL_COMMAND ""
    )
    add_dependencies(dft libxsmm)
  endif()
endif()
endif() # DISABLE_DFT

if(ENABLE_FFTW AND BUILD_FFTW)
#  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/include/fftw3.h")
    ExternalProject_Add(libfftw3
      URL https://www.fftw.org/fftw-3.3.10.tar.gz
      PREFIX ${PROJECT_BINARY_DIR}/deps
      INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
      BUILD_IN_SOURCE True
      CONFIGURE_COMMAND ./configure --enable-static=no --enable-shared=yes --enable-threads CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER} prefix=<INSTALL_DIR>
      BUILD_COMMAND make -j4 install
    )
    add_dependencies(fft libfftw3)
    add_dependencies(pbc libfftw3)
#  endif()
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
