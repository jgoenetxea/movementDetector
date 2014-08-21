//
//  MovementDetector.h
//  movementDetection
//
//  Created by Jon Goenetxea on 21/08/14.
//
//

#ifndef __movementDetection__MovementDetector__
#define __movementDetection__MovementDetector__

#include <iostream>

#include <opencv2/core/core.hpp>


class MovementDetector
{
public:
    MovementDetector();
    ~MovementDetector();
    
    void reset();
    bool addImage(const cv::Mat& greyImage);
    void getOutlierMask( cv::Mat& outlierMask );
    void getMeanImage( cv::Mat& meanImage );
    
    bool wasMovement( const float changedPixelPercentage = 0.2f );
    
private:
    int __step_counter;
    int __last_outlier_ammount;
    float __min_steps_for_deviation_update;
    float __outlier_Threshold;
    float __outliers_percent_threshold;
    float __alpha;
    cv::Mat __mean, __stdev, __moment_2, __coe, __outlier, __normalGreyImage;
    cv::Mat __transitionImage;
    
    void normalizeImage(const cv::Mat& src_img, cv::Mat& dst_img);
};

#endif /* defined(__movementDetection__MovementDetector__) */
