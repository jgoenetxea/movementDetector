cmake_minimum_required( VERSION 2.8 )
project( movementDetection )

find_package( OpenCV REQUIRED )

set( movementDetection_SRCS
	main.cpp
    MovementDetector.cpp
    MovementDetector.h
    TextureUpdate.cpp
    TextureUpdate.h
)

IF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)
	#Include OpenCV var in Android
	SET(FACETR_OPENCV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../opencv/android/OpenCV-2.4.8-android-sdk/sdk/native/jni/include)
	INCLUDE_DIRECTORIES(${FACETR_OPENCV_INCLUDE_DIR})
	
ENDIF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)

IF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)
  link_directories( ${NITO_PATH_MAIN}/proj.android/libs/armeabi-v7a/ )
ELSEIF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)
  link_directories( ${CMAKE_BINARY_DIR}/bin )
ENDIF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)

add_executable( movementDetection ${movementDetection_SRCS} )
target_link_libraries( movementDetection ${OpenCV_LIBRARIES} )