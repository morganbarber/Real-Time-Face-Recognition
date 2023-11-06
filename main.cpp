#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
    CascadeClassifier faceCascade;
    faceCascade.load("Cascades/haarcascade_frontalface_default.xml");

    VideoCapture capture(0);
    capture.set(CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CAP_PROP_FRAME_HEIGHT, 480);

    if(!capture.isOpened())
    {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    Mat img;
    while(true)
    {
        capture.read(img);
        
        if(img.empty())
        {
            cout << "Failed to capture an image" << endl;
            return -1;
        }

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.2, 5, 0|CASCADE_SCALE_IMAGE, Size(20, 20));

        for ( size_t i = 0; i < faces.size(); i++ )
        {
            rectangle( img, faces[i], Scalar(255,0,0), 2);
            Mat roi_gray = gray(faces[i]);
            Mat roi_color = img(faces[i]);
        
            // To manually release memory
            roi_gray.release();
            roi_color.release();
        }

        imshow("video", img);
        char c = (char)waitKey(30);
        if( c == 27 )
        { 
            break;
        }

        gray.release();
    }

    capture.release();
    img.release();
    destroyAllWindows();

    return 0;
}
