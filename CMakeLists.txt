cmake_minimum_required(VERSION 2.4.6)

include_directories(src external)

find_package(Boost REQUIRED COMPONENTS thread system)
find_package(OpenCV REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_library(fzht_superpix
  external/FzHtSuperpix/src/disjoint-set.cc
  external/FzHtSuperpix/src/segment-graph.cc
  external/FzHtSuperpix/src/segment-image.cc
  external/FzHtSuperpix/src/filter.cc)

add_library(mrf_image_classifier
  src/features/SuperpixelBlobber.cc 
  src/features/FeatureMatrix.cc 
  src/objective/BinarySubmodularObjective.cc  
  src/objective/BinarySubmodularEnergy.cc  
  src/objective/GraphCutMinimizer.cc 
  src/classifier/BinarySubmodularImageClassifier.cc 
  src/classifier/NaryImageClassifier.cc 
  src/util/Dvec.cc 
  src/util/uBLASUtils.cc)

set(projectLibs 
  mrf_image_classifier
  fzht_superpix
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)                

add_executable(binaryTrainBatch src/input/binaryTrainBatch.cc)
target_link_libraries(binaryTrainBatch ${projectLibs})

add_executable(binaryEvalBatch src/input/binaryEvalBatch.cc)
target_link_libraries(binaryEvalBatch ${projectLibs})

add_executable(segmentImages src/input/segmentImages.cc)
target_link_libraries(segmentImages ${projectLibs})

#set_target_properties(mrf_image_classifier
#  PROPERTIES
#  COMPILE_FLAGS "-O2 -DNDEBUG"
#  LINK_FLAGS "-L/opt/boost/lib -lboost_thread-gcc42-mt-1_35")

# -DNDEBUG is critical for remotely decent ublas performance
# FIXME: hack! ROS boost package broken!
#set_target_properties(mrf_image_classifier
#  PROPERTIES
#  COMPILE_FLAGS "-O2 -DNDEBUG"
#  LINK_FLAGS "-L/opt/boost/lib -lboost_thread-gcc42-mt-1_35")
