#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

string filepath = "/home/lqx/folder/SVMTrain/ImageSet/";

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
                //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
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

void preprocess(Mat input, Mat &bgr_output, Mat &hsv_output)
{
    gammaCorrection(input, bgr_output, 0.75);
    cv::GaussianBlur(input, bgr_output, cv::Size(5, 5), 3, 3);

    cvtColor(bgr_output, hsv_output, CV_BGR2HSV);
}

int main()
{
    VideoCapture videoCapture(2);
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    if (!videoCapture.isOpened())
    {
        cout << "Video not open!" << endl;
        return 1;
    }
    Mat frame;
    Mat bgr_frame;
    Mat hsv_frame;

    int index = 1;
    while (videoCapture.isOpened())
    {
        bool ret = videoCapture.read(frame);

        if (ret)
        {
            resize(frame, frame, Size(480, 360));
            preprocess(frame, bgr_frame, hsv_frame);

            Mat image;
            hconcat(bgr_frame, hsv_frame, image);

            imshow("frame", image);

            char c = waitKey(30); //延时30毫秒
            if (c == 27) break;
            else if (c == ' ')
            {
                imwrite(filepath + "bgr/" + std::to_string(index) + "bgr.jpg", bgr_frame);
//                imwrite(filepath + "hsv/" + std::to_string(index) + "hsv.jpg", hsv_frame);
                cout << "Saving:\t" << index << endl;
                index++;
            }
        }
        else break;
    }
    cout << "FINISH ALL WORK----------" << endl;
    videoCapture.release(); //when everything done, release the capture
    destroyAllWindows();
    return 0;
}