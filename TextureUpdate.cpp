//
//  TextureUpdate.cpp
//  textureUpdate
//
//  Created by Jon Goenetxea on 21/08/14.
//
//

#include "TextureUpdate.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#define INITIAL_SIGMA          0.1f


TextureUpdate::TextureUpdate()
{
    __step_counter = 0;
    __min_steps_for_deviation_update = 40.0f;
    __outlier_Threshold = 3.f;
    __outliers_percent_threshold = 20.0f;
    __alpha = 0.007f;
}

TextureUpdate::~TextureUpdate()
{
    reset();
}

void TextureUpdate::reset()
{
    __step_counter = 0;
    __mean.release();
    __stdev.release();
    __coe.release();
    __moment_2.release();
    __outlier.release();
}

void TextureUpdate::getOutlierMask( cv::Mat& outlierMask )
{
    double minV, maxV;
    minMaxLoc(__outlier, &minV, &maxV);
    __outlier.convertTo(__transitionImage, cv::DataType<unsigned char>::type , 255/(maxV-minV), -minV*255/(maxV-minV));
    
    __transitionImage.copyTo(outlierMask);
}

void TextureUpdate::getMeanImage( cv::Mat& meanImage )
{
    double minV, maxV;
    minMaxLoc(__mean, &minV, &maxV);
    __mean.convertTo(__transitionImage, cv::DataType<unsigned char>::type , 255/(maxV-minV), -minV*255/(maxV-minV));
    
    __transitionImage.copyTo(meanImage);
}

bool TextureUpdate::addImage(const cv::Mat& greyImage)
{    
    int n_outlier = 0;
    bool oclussion_occurred = false;
    
    normalizeImage(greyImage, __normalGreyImage);
    
    if(__step_counter == 0)
    {
        // Iniciamos el vector media
        __mean.release();
        __mean      = cv::Mat::zeros(__normalGreyImage.size(), CV_32F);
        __stdev.release();
        __stdev     = cv::Mat::ones(__normalGreyImage.size(), CV_32F);
        __coe.release();
        __coe       = cv::Mat::ones(__normalGreyImage.size(), CV_32F);
        __outlier.release();
        __outlier   = cv::Mat::zeros(__normalGreyImage.size(), CV_32F);
        __moment_2.release();
        __moment_2  = cv::Mat::zeros(__normalGreyImage.size(), CV_32F);
        
        float init_sigma_2 = INITIAL_SIGMA * INITIAL_SIGMA;
        
        __mean = __normalGreyImage.clone();    // Si se clona, no hace falta inicializar arriba
        // Iniciar el momento para calcular la varianza despues
        __moment_2 = __normalGreyImage.mul(__normalGreyImage);
        __stdev = init_sigma_2 * __stdev;
    }
    else if(__step_counter < __min_steps_for_deviation_update)
    {
        // Calc the local alpha value for the first "__min_steps_for_deviation_update" frames
        float alpha = 0.1f;
        
        for(int i=0 , n=__normalGreyImage.cols ; i<n ; ++i)
        {
            for(int j=0 , m=__normalGreyImage.rows ; j<m ; ++j)
            {
                __moment_2.at<float>(j, i) = __moment_2.at<float>(j, i) + (__normalGreyImage.at<float>(j, i) * __normalGreyImage.at<float>(j, i));
                __stdev.at<float>(j, i) = (__moment_2.at<float>(j, i) / (float)__step_counter+1) - (__mean.at<float>(j, i) * __mean.at<float>(j, i));
                
                if (  __stdev.at<float>(j, i) < 0.01)
                {
                    __stdev.at<float>(j, i) = 0.01f; // Evitamos los valores negativos y ceros
                }
                
                __mean.at<float>(j, i) = ( (1-alpha) * __mean.at<float>(j, i) ) + ( alpha * __normalGreyImage.at<float>(j, i) );
                __coe.at<float>(j, i) = 1.0f/__stdev.at<float>(j, i);
                
                // TODO: Quitar función de cálculo de outliers
                __outlier.at<float>(j, i) = 0;
                if(fabs(__normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i))/sqrt(__stdev.at<float>(j, i)) >= __outlier_Threshold)
                {
                    __outlier.at<float>(j, i) = 1;
                }
            }
        }
    }
    else
    {
        float txt_diff = 0.f;
        for(int i=0 , n=__normalGreyImage.cols ; i<n ; ++i)
        {
            for(int j=0 , m=__normalGreyImage.rows ; j<m ; ++j)
            {
                if ( fabs(__normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i)) / sqrt(__stdev.at<float>(j, i)) < __outlier_Threshold )
                {
                    txt_diff = (__normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i));
                    //__stdev.at<float>(j, i) = ( (1-__alpha)*__stdev.at<float>(j, i) ) + (__alpha * txt_diff * txt_diff);
                    __stdev.at<float>(j, i) = ( (1-__alpha) * __stdev.at<float>(j, i) ) + (__alpha * (__normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i)) * (__normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i)));
                    if (  __stdev.at<float>(j, i) < 0.01) __stdev.at<float>(j, i) = 0.01f; // Evitamos los valores negativos y ceros
                    __mean.at<float>(j, i) = ( (1-__alpha) * __mean.at<float>(j, i) ) + ( __alpha * __normalGreyImage.at<float>(j, i) );
                }
                else
                {
                    n_outlier++;
                }
                // Cálculo del coeficiente para el error etc...
                __coe.at<float>(j, i) = 1.0f / __stdev.at<float>(j, i);
                
                // TODO: Quitar función de cálculo de outliers
                __outlier.at<float>(j, i) = 0;
                if( fabs( __normalGreyImage.at<float>(j, i) - __mean.at<float>(j, i))/sqrt(__stdev.at<float>(j, i) ) >= __outlier_Threshold )
                {
                    __outlier.at<float>(j, i) = 1;
                }
            }
        }
        // Detectar la oclusión si tenemos muchos outliers
        if(n_outlier * 100 / (__normalGreyImage.rows*__normalGreyImage.cols) > __outliers_percent_threshold)
        {
            oclussion_occurred = true;
            std::cout << "Oclusion detected!!!" << std::endl;
        }
    }
    
    if(__step_counter == (int)__min_steps_for_deviation_update)
    {
        std::cout << "Pasamos a modo actualización continua..." << std::endl;
    }
    
    __step_counter++;
    
    return oclussion_occurred;
}

void TextureUpdate::normalizeImage(const cv::Mat& src_img, cv::Mat& dst_img)
{
    /* Normalise the patch Mask_previous */
    cv::Scalar meanValue, stdDevValue;
    meanStdDev( src_img, meanValue, stdDevValue );
    src_img.convertTo( dst_img, CV_32F, 1/stdDevValue.val[0], -meanValue.val[0]/stdDevValue.val[0] );
}
