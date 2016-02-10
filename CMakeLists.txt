cmake_minimum_required(VERSION 2.8.9)
  project(test_sup)

  # Tell CMake to run moc when necessary:
  set(CMAKE_AUTOMOC ON)
  # As moc files are generated in the binary dir, tell CMake
  # to always look for includes there:
  set(CMAKE_INCLUDE_CURRENT_DIR ON)


  ################################
  # PATH vlFeat: http://www.vlfeat.org/download.html
  ################################

  find_path(VLFEAT_INCLUDE_DIR NAMES vl/generic.h
          PATH_SUFFIXES vl)

	
  set(VLFEAT_INCLUDE_DIRS ${VLFEAT_INCLUDE_DIR} )
  set(VLFEAT_LIBRARIES ${VLFEAT_INCLUDE_DIR}/bin/maci64)


  ##################################################################################
  #   PATH  OpenCV
  ##################################################################################
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDES})

  ##################################################################################
  #   PATH  CAFFE, cuda, blas
  ##################################################################################
  set(CAFFE_ROOT /Users/acambra/caffe-master/distribute)

  include_directories(${CAFFE_ROOT}/include)
  include_directories(/Developer/NVIDIA/CUDA-7.5/include)
  include_directories(/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers)

  # libs

  set (CMAKE_FLAGS_CAFFE "-Wno-deprecated-declarations -Wno-missing-braces")
  set (caffe_LIBS ${CAFFE_ROOT}/lib/libcaffe.so)
  set (glog_LIBS /usr/local/lib/libglog.dylib)
  set (protobuf_LIBS /usr/local/lib/libprotobuf.dylib)


  ##################################################################################
  #     Compiler FLAGS
  ##################################################################################
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=gnu++11 -stdlib=libc++ ${CMAKE_FLAGS_CAFFE}")

  ##################################################################################
  #     paths: include and lib user
  ##################################################################################

  include_directories(libs)
  include_directories(libs/lib_slic)
  include_directories(${VLFEAT_INCLUDE_DIRS})


  find_package(Boost COMPONENTS system filesystem program_options REQUIRED)


  ##################################################################################
  #     src
  ##################################################################################
  add_library(slic libs/lib_slic/SLIC.cpp)

  add_executable(${PROJECT_NAME}  
        src/superpixel.cpp
        src/superpixels.cpp
        src/labelSet.cpp
        src/utilsCaffe.cpp
        src/test_superpixels.cpp)



target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${VLFEAT_LIBRARIES}/libvl.dylib slic ${Boost_LIBRARIES} ${caffe_LIBS} ${glog_LIBS} ${protobuf_LIBS})

  file(COPY ${VLFEAT_LIBRARIES}/libvl.dylib DESTINATION ${CMAKE_BINARY_DIR}/Debug)
  file(COPY ${CAFFE_ROOT}/lib/libcaffe.so DESTINATION ${CMAKE_BINARY_DIR}/Debug)


