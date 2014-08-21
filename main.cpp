#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
#include <fstream>

#include "TextureUpdate.h"
#include "MovementDetector.h"

using namespace std;
using namespace cv;

//#define USE_MD 1

string inputVideoName = "";//"F:/Vicomtech/Databases/iPad/IMG_0039.MOV";
int inputType = CV_CAP_DSHOW; // CV_CAP_OPENNI;CV_CAP_DSHOW

bool paused = false, detect = false, fullReset = true, track = false;
Mat frame, prevFrame, frame_gray, frame_gray_downscaled, frame_color_downscaled;
int normWidth = 320;
int downscale = 6;

bool flipped = false;

bool video = false;
VideoWriter vw( "output.avi", CV_FOURCC('D','I','V','X'), 30, frame.size() );

#ifdef USE_MD
MovementDetector md;
#else
TextureUpdate tu;
#endif

cv::Mat outlierMask, meanImage;

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
            
			// Detect facial parts
			if( !frame.empty() )
			{
				cvtColor( frame, frame_gray, CV_BGR2GRAY );
				resize( frame_gray, frame_gray_downscaled, Size(frame_gray.cols/downscale,frame_gray.rows/downscale) );
                
#ifdef USE_MD
                // Process frame with detector
                md.addImage(frame_gray_downscaled);
                md.getOutlierMask(outlierMask);
                md.getMeanImage(meanImage);
                if( md.wasMovement() )
                {
                    rectangle(frame, cv::Point( 0, 0 ), cv::Point( frame.cols-1, frame.rows-1 ), Scalar(0, 0, 255), 2 );
                }
#else
                // Process frame with detector
                tu.addImage(frame_gray_downscaled);
                tu.getOutlierMask(outlierMask);
                tu.getMeanImage(meanImage);
#endif
			}
            
			else
			{
				cout << " --(!) No captured frame -- Break!" << endl;
				break;
			}
            
            // Draw framerate on display image
			stringstream ss;
			ss << (int)floor(fps) << " frames/sec";
			putText( frame, ss.str(), Point(10,frame.rows - 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,255,255) );
            
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
            addImageInPosition( frame, meanImage, cv::Point(frame_gray_downscaled.cols, 0) );
            addImageInPosition( frame, outlierMask, cv::Point(frame_gray_downscaled.cols*2, 0) );
            
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
                    
                case 'r':
                case 'R':
#ifdef USE_MD
                    // Process frame with detector
                    md.reset();
#else
                    // Process frame with detector
                    tu.reset();
#endif
                    break;
                    
                default:
                    ;
			}
		}
	}
    
	destroyAllWindows();

	return 0;
}


