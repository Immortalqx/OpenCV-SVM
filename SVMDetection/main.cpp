#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

bool bgr_failed = true;

void gammaCorrection(Mat &src, Mat &dst, float fGamma)
{
    CV_Assert(src.data);

    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));

    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = saturate_cast<uchar>(pow((float) (i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
        case 1:
        {

            MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
//                *it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                *it = lut[(*it)];

            break;
        }
        case 3:
        {
            MatIterator_<Vec3b> it, end;
            for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
            {
                //(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
                //(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
                //(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }
            break;
        }
    }
}

//void preprocess(Mat input, Mat &bgr_output, Mat &hsv_output)
void preprocess(const Mat &input, Mat &bgr_output)
{
//    gammaCorrection(input, bgr_output, 0.75);

    cv::GaussianBlur(input, bgr_output, cv::Size(5, 5), 3, 3);
}

void runDetectMultiScale(const HOGDescriptor &hog, Mat img, const String &type)
{
    if (type == "bgr")
        bgr_failed = true;

    vector<Rect> detections;
    vector<double> weights;

    hog.detectMultiScale(img, detections, weights, 0, Size(), Size(), 1.25, 8.0);
    for (size_t j = 0; j < detections.size(); j++)
    {
        if (weights[j] < 0.43) continue;
        if (type == "bgr")
            bgr_failed = false;
        Scalar color = Scalar(0, weights[j] * weights[j] * 400, 0);
        rectangle(img, detections[j], color, img.cols / 400 + 1);
    }
}

void test_trained_detector(const String &obj_det_filename, const String &index)
{
    cout << "Testing trained detector..." << endl;

    HOGDescriptor bgr_hog;
    bgr_hog.load(obj_det_filename + "RM256x256" + ".xml");
    string bgr_window = "BGR_Detection";

    int delay = 0;
    int width = 640;
    int height = 480;
    int fps = 30;

    VideoCapture cap;
    cap.open(2);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    cap.set(CV_CAP_PROP_FPS, fps);

    for (int i = 0;; i++)
    {
        Mat img;
        Mat bgr_img;

        cap >> img;
//        resize(img, img, Size(1280, 960));
        delay = 30;
        if (img.empty())
        {
            return;
        }

        // Start timer
        double timer = (double) getTickCount();

        preprocess(img, bgr_img);

        runDetectMultiScale(bgr_hog, bgr_img, "bgr");

        // Calculate Frames per second (FPS)
        double fpsx = getTickFrequency() / ((double) getTickCount() - timer);

        Mat saved = bgr_img;

        putText(saved, "SVM", Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
        putText(saved, "FPS : " + to_string(int(fpsx)), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50),
                2);
        if (bgr_failed)
            putText(saved, "Detection failed", Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        imshow("RESULT", saved);

        if (waitKey(delay) == ' ')
        {
            break;
        }
    }
}

int main(int argc, char **argv)
{

    test_trained_detector("/home/lqx/folder/SVMTrain/HOGpedestrian_", "2");

    return 0;
}