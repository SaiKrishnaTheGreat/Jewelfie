//created by sdyavarasetti on 03/12/2017.

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace cv;
using namespace std;

char ch;

void show(String winname,Mat &frame)
{
    namedWindow(winname,WINDOW_NORMAL);
    imshow(winname,frame);
    waitKey( 30 );
}

int main(int argc, char* argv[])
{
    string faceCascadeName = argv[1];
    CascadeClassifier faceCascade;
    
    if( !faceCascade.load(faceCascadeName) )
    {
        cerr << "Error loading cascade file. Exiting!" << endl;
        return -1;
    }
    
    Mat faceMask = imread(argv[2]);
    show("0.faceMask",faceMask);
    
    if ( !faceMask.data )
    {
        cerr << "Error loading mask image. Exiting!" << endl;
    }
    
    // Current frame
    Mat frame, frameGray;
    Mat frameROI, faceMaskSmall;
    Mat grayMaskSmall, grayMaskSmallThresh, grayMaskSmallThreshInv;
    Mat maskedFace, maskedFrame;
    
    //char ch;
    
    // Create the capture object
    // 0 -> input arg that specifies it should take the input from the webcam
    VideoCapture cap(0);
    
    // If you cannot open the webcam, stop the execution!
    if( !cap.isOpened() )
        return -1;
    
    //create GUI windows
    //namedWindow("Frame",WINDOW_NORMAL);
    
    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;
    
    vector<Rect> faces;
    
    // Iterate until the user presses the Esc key
    while(true)
    {
        // Capture the current frame
        cap >> frame;
        //show("1.Frame",frame);
        
        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
        //show("2.ResizedFrame",frame);
        
        // Convert to grayscale
        cvtColor(frame, frameGray, CV_BGR2GRAY);
        //show("3.GrayFrame",frameGray);
        
        // Equalize the histogram
        equalizeHist(frameGray, frameGray);
        //show("4.EqualizeFrame",frameGray);
        
        // Detect faces
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        
        // Draw green rectangle around the face
        for(int i = 0; i < faces.size(); i++)
        {
            Rect faceRect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
            
            // Custom parameters to make the mask fit your face. You may have to play around with them to make sure it works.
            int x = faces[i].x - int(0.1*faces[i].width);
            int y = faces[i].y - int(0.0*faces[i].height);
            int w = int(1.1 * faces[i].width);
            int h = int(1.3 * faces[i].height);
            
            // Extract region of interest (ROI) covering your face
            frameROI = frame(Rect(x,y+h,w,h));
            show("5.frameROI",frameROI);
            
            // Resize the face mask image based on the dimensions of the above ROI
            resize(faceMask, faceMaskSmall, Size(w,h));
            show("6.ResizedMask",faceMaskSmall);
            
            // Convert the above image to grayscale
            cvtColor(faceMaskSmall, grayMaskSmall, CV_BGR2GRAY);
            show("7.GrayMask",grayMaskSmall);
            
            // Threshold the above image to isolate the pixels associated only with the face mask
            threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, CV_THRESH_BINARY_INV);
            show("8.ThresholdMask",grayMaskSmallThresh);
            // Create mask by inverting the above image (because we don't want the background to affect the overlay)
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv);
            show("9.bitwise_not-InvThresholdMask",grayMaskSmallThreshInv);
            
            // Use bitwise "AND" operator to extract precise boundary of face mask
            bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh);
            show("10.bitwise_and-1--->TO get Jewel",maskedFace);
            
            // Use bitwise "AND" operator to overlay face mask
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv);
            show("11.bitwise_and-2---->To get faceROI+Jewel",maskedFrame);
            
            // Add the above masked images and place it in the original frame ROI to create the final image
            add(maskedFace, maskedFrame, frame(Rect(x,y+h,w,h)));
        }
        
        // Show the current frame
        show("12.Final-Frame", frame);
        
        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey( 30 );
        if (ch == 27) {
            break;
        }
    }
    
    // Release the video capture object
    cap.release();
    
    // Close all windows
    destroyAllWindows();
    
    return 1;
}