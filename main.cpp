#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

string inputVideoName = "";//"F:/Vicomtech/Databases/iPad/IMG_0039.MOV";
int inputType = CV_CAP_DSHOW; // CV_CAP_OPENNI;CV_CAP_DSHOW

bool paused = false, detect = false, fullReset = true, track = false;
Mat frame, prevFrame, frame_gray, frame_gray_downscaled, frame_color_downscaled;
int normWidth = 320;
int downscale = 6;

bool flipped = false;

bool video = false;
VideoWriter vw( "output.avi", CV_FOURCC('D','I','V','X'), 30, frame.size() );


void addImageInRegion( const cv::Mat& mainImage, const cv::Mat& smallImage, const cv::Rect location )
{
    Mat imagerect = mainImage(cv::Rect(0,0,frame_gray_downscaled.cols, frame_gray_downscaled.rows));
    imshow("rect", imagerect);
    cvtColor( frame_gray_downscaled, frame_color_downscaled, CV_GRAY2RGB );
    frame_color_downscaled.copyTo(imagerect);
}

cv::Mat coloredSmallImage;
void addImageInPosition( const cv::Mat& mainImage, const cv::Mat& smallImage, const cv::Point position )
{
    Mat imagerect = mainImage(cv::Rect( position.x, position.y, smallImage.cols,  smallImage.rows ));
    if(smallImage.channels() == 1)
    {
        cvtColor( smallImage, coloredSmallImage, CV_GRAY2RGB );
        coloredSmallImage.copyTo(imagerect);
    }
    else
    {
        smallImage.copyTo(imagerect);
    }
}

// Main function
int main( int argc, const char **argv )
{
	VideoCapture capture;
	if( inputVideoName == "" )
    {
#ifdef __APPLE__
        capture.open( 0 );
#elif defined(__WINDOWS__)
        capture.open( CV_CAP_DSHOW );
#else
        capture.open( 0 );
#endif
        // Define the camera image size
        capture.set(CV_CAP_PROP_FRAME_WIDTH,800);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT,600);
    }
    else
    {
        capture.open( inputVideoName );
    }
    
    // This block is to wait the camera activation
    Mat bigFrame;
    int cont = 0;
    if( capture.isOpened() )
    {
        printf("Wait until the camera is completely loaded\n");
        bool cameraIsLoaded = false;
        while(!cameraIsLoaded)
        {
            cont += 1;
            capture.read(bigFrame);
            
            if(!bigFrame.empty())
            {
                printf("OK!, The camera is working now, so follow the execution\n");
                cameraIsLoaded = true;
            }
            
            int key = cv::waitKey(10);
            
            if (key == 27 )
            {
                printf("Well, you want to exit before the camere is ready!\n");
                exit(0);
            }
        }
    }
    printf("Camera loaded in %d frames\n", cont);
    printf("Frame data: [%dx%d]\n", bigFrame.cols, bigFrame.rows);

	// Main loop
	if( capture.isOpened() )
	{
		int64 t1, t0 = cvGetTickCount();
		int fnum = 0;
		float fps = 0;
        
		while( true )
		{
			// Grab image
			if( !paused )
			{
				capture.grab();
				capture.retrieve( frame );
				if( flipped ) flip( frame, frame, 1 );
				frame.copyTo( prevFrame );
			}
            
			else prevFrame.copyTo( frame );
            
			// Draw framerate on display image
			stringstream ss;
			ss << (int)floor(fps) << " frames/sec";
			putText( frame, ss.str(), Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,255,255) );
            
			// Detect facial parts
			if( !frame.empty() )
			{
				cvtColor( frame, frame_gray, CV_BGR2GRAY );
				resize( frame_gray, frame_gray_downscaled, Size(frame_gray.cols/downscale,frame_gray.rows/downscale) );
			}
            
			else
			{
				cout << " --(!) No captured frame -- Break!" << endl;
				break;
			}
            
			// Calculate framerate
			if( fnum >= 9 )
			{
				t1 = cvGetTickCount();
				fps = 10.f / ( (float)( (t1-t0)/cvGetTickFrequency() ) / (float)1e+6 );
				t0 = t1;
				fnum = 0;
			}
			else
            {
                fnum += 1;
            }
            
            // Add grey image to main image
//            Mat imagerect = frame(cv::Rect(0,0,frame_gray_downscaled.cols, frame_gray_downscaled.rows));
//            imshow("rect", imagerect);
//            cvtColor( frame_gray_downscaled, frame_color_downscaled, CV_GRAY2RGB );
//            frame_color_downscaled.copyTo(imagerect);
            addImageInPosition( frame, frame_gray_downscaled, cv::Point( 0, 0 ) );
            
            // Show frame
			imshow( "Detection Result", frame );
            
            // Manage keyboard
			int c = waitKey( 10 );
			if( (char)c == 27 )
			{
				destroyAllWindows();
                
				break;
			}
            
			switch( c )
			{
                case 'f':
                case 'F':
                    flipped = !flipped;
                    break;
                    
                case 'p':
                case 'P':
                    paused = !paused;
                    break;
                    
                default:
                    ;
			}
		}
	}
    
	destroyAllWindows();

	return 0;
}


#define CALC_OUTLIER_MASK 1
#define CALC_OUTLIERS 1
#define INITIAL_SIGMA          0.1f
cv::Mat __mean, __stdev, __moment_2, __coe, __outlier;
int __step_counter = 0;
float __min_steps_for_deviation_update = 40.0f;
float __outlier_Threshold = 3.f;
float __outliers_percent_threshold = 20.0f;
float __alpha = 0.007f;

bool updateGaussianData(const cv::Mat& currentTexture){
    
    int n_outlier = 0;
    bool oclussion_occurred = false;
    if(__step_counter == 0)
    {
        // Iniciamos el vector media
        __mean.release();       __mean      = cv::Mat::zeros(currentTexture.size(), CV_32F);
        __stdev.release();      __stdev     = cv::Mat::ones(currentTexture.size(), CV_32F);
        __coe.release();        __coe       = cv::Mat::ones(currentTexture.size(), CV_32F);
        if(CALC_OUTLIER_MASK) { __outlier.release();    __outlier   = cv::Mat::zeros(currentTexture.size(), CV_32F); }
        __moment_2.release();   __moment_2  = cv::Mat::zeros(currentTexture.size(), CV_32F);
        float init_sigma_2 = INITIAL_SIGMA*INITIAL_SIGMA;
        
        __mean = currentTexture.clone();    // Si se clona, no hace falta inicializar arriba
        // Iniciar el momento para calcular la varianza despues
        __moment_2 = currentTexture.mul(currentTexture);
        __stdev = init_sigma_2*__stdev;
    }
    else if(__step_counter < __min_steps_for_deviation_update)
    {
        // Calc the local alpha value for the first "__min_steps_for_deviation_update" frames
        float alpha = 0.1f;
        
        for(int i=0 , n=currentTexture.rows ; i<n ;++i){
            __moment_2.at<float>(i) = __moment_2.at<float>(i) + (currentTexture.at<float>(i) * currentTexture.at<float>(i));
            __stdev.at<float>(i) = (__moment_2.at<float>(i) / (float)__step_counter+1) - (__mean.at<float>(i) * __mean.at<float>(i));
            
            if (  __stdev.at<float>(i) < 0.01)
                __stdev.at<float>(i) = 0.01f; // Evitamos los valores negativos y ceros
            
            __mean.at<float>(i) = ( (1-alpha) * __mean.at<float>(i) ) + ( alpha * currentTexture.at<float>(i) );
            __coe.at<float>(i) = 1.0f/__stdev.at<float>(i);
            
            // TODO: Quitar función de cálculo de outliers
            if(CALC_OUTLIER_MASK){
                __outlier.at<float>(i) = 0;
                if(fabs(currentTexture.at<float>(i) - __mean.at<float>(i))/sqrt(__stdev.at<float>(i)) >= __outlier_Threshold){
                    __outlier.at<float>(i) = 1;
                }
            }
        }
    }
    else
    {
        float txt_diff = 0.f;
        for(int i=0 , n=currentTexture.rows ; i<n ;++i){
            if ( !CALC_OUTLIERS || fabs(currentTexture.at<float>(i) - __mean.at<float>(i))/sqrt(__stdev.at<float>(i)) < __outlier_Threshold ){
                txt_diff = (currentTexture.at<float>(i) - __mean.at<float>(i));
                //__stdev.at<float>(i) = ( (1-__alpha)*__stdev.at<float>(i) ) + (__alpha * txt_diff * txt_diff);
                __stdev.at<float>(i) = ( (1-__alpha)*__stdev.at<float>(i) ) + (__alpha * (currentTexture.at<float>(i) - __mean.at<float>(i)) * (currentTexture.at<float>(i) - __mean.at<float>(i)));
                if (  __stdev.at<float>(i) < 0.01) __stdev.at<float>(i) = 0.01f; // Evitamos los valores negativos y ceros
                __mean.at<float>(i) = ( (1-__alpha)*__mean.at<float>(i) ) + ( __alpha * currentTexture.at<float>(i) );
            }else{
                n_outlier++;
            }
            // Cálculo del coeficiente para el error etc...
            __coe.at<float>(i) = 1.0f/__stdev.at<float>(i);
            
            // TODO: Quitar función de cálculo de outliers
            if(CALC_OUTLIER_MASK){
                __outlier.at<float>(i) = 0;
                if(fabs(currentTexture.at<float>(i) - __mean.at<float>(i))/sqrt(__stdev.at<float>(i)) >= __outlier_Threshold){
                    __outlier.at<float>(i) = 1;
                }
            }
        }
        // Detectar la oclusión si tenemos muchos outliers
        if(CALC_OUTLIERS && (n_outlier * 100 / currentTexture.rows > __outliers_percent_threshold)){
            oclussion_occurred = true;
            std::cout << "Oclusion detected!!!" << std::endl;
        }
    }
    
    if(__step_counter == (int)__min_steps_for_deviation_update)
    {
        std::cout << "Pasamos a modo actualización continua..." << std::endl;
    }
    
    return oclussion_occurred;
}

