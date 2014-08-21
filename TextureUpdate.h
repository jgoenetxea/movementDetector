//
//  TextureUpdate.h
//  movementDetection
//
//  Created by Jon Goenetxea on 21/08/14.
//
//

#ifndef __textureUpdate__TextureUpdate__
#define __textureUpdate__TextureUpdate__

#include <iostream>

#include <opencv2/core/core.hpp>


class TextureUpdate
{
public:
    TextureUpdate();
    ~TextureUpdate();
    
    void reset();
    bool addImage(const cv::Mat& greyImage);
    void getOutlierMask( cv::Mat& outlierMask );
    void getMeanImage( cv::Mat& meanImage );
    
private:
    int __step_counter;
    float __min_steps_for_deviation_update;
    float __outlier_Threshold;
    float __outliers_percent_threshold;
    float __alpha;
    cv::Mat __mean, __stdev, __moment_2, __coe, __outlier, __normalGreyImage;
    cv::Mat __transitionImage;
    
    void normalizeImage(const cv::Mat& src_img, cv::Mat& dst_img);
};

#endif /* defined(__movementDetection__TextureUpdate__) */
