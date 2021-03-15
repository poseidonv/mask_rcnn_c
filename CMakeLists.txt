cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
project(torchscript_traced_mask_rcnn)

find_package(OpenCV REQUIRED)
set(CMAKE_PREFIX_PATH
	/home/poseidon/Downloads/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_PREFIX_PATH
/home/poseidon/anaconda3/envs/Segment/vision-0.8.1-rc1/torchvision)
find_package(TorchVision REQUIRED)
#find_package(gflags REQUIRED)

#add_executable(caffe2_mask_rcnn caffe2_mask_rcnn.cpp)
#target_link_libraries(
  #caffe2_mask_rcnn
  #"${TORCH_LIBRARIES}" gflags glog protobuf ${OpenCV_LIBS})
#set_property(TARGET caffe2_mask_rcnn PROPERTY CXX_STANDARD 14)


add_executable(torchscript_traced_mask_rcnn torchscript_traced_mask_rcnn.cpp)
target_compile_features(torchscript_traced_mask_rcnn PUBLIC cxx_range_for)
target_link_libraries(
  torchscript_traced_mask_rcnn
  "${TORCH_LIBRARIES}" ${OpenCV_LIBS} TorchVision::TorchVision)
# set_property(TARGET torchscript_traced_mask_rcnn PROPERTY CXX_STANDARD 11)
