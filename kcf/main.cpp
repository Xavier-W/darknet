#include <iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/opencv.hpp>
#include<time.h>
#include "../src/network.h"
#include "../include/darknet.h"
#include "KCFcpp//src/improcess.hpp"
#include "KCFcpp//src/kcftracker.hpp"
#include "../src/parser.h"

#define USE_YOLO
// #define USE_SSD

using namespace std;
using namespace cv;
using namespace cv::dnn;


int main()
{
    VideoCapture cap("../../../../cars.mp4");//读取视频，请自行修改相应路径
    Mat frame;

    #if defined(USE_SSD)
    cap.read(frame);
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    const std::string caffeConfigFile = "../deploy.prototxt";
    const std::string caffeWeightFile = "../res10_300x300_ssd_iter_140000_fp16.caffemodel";
    Net net_caffe = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net_caffe.setPreferableBackend(DNN_BACKEND_DEFAULT);
    #elif defined(USE_YOLO)
    string cfgfile = "../../cfg/yolov4-tiny.cfg";//读取模型文件，请自行修改相应路径
    string weightfile = "../../yolov4-tiny.weights";
    float thresh=0.5;//参数设置
    float nms=0.35;
    int classes=80;

    network *net=load_network((char*)cfgfile.c_str(),(char*)weightfile.c_str(),0);//加载网络模型
    set_batch_network(net, 1);

    Mat rgbImg;
    vector<string> classNamesVec;
    ifstream classNamesFile("../../cfg/coco.names");//标签文件coco有80类

    if (classNamesFile.is_open()){
        string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    bool stop=false;
    #endif

    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const float confidenceThreshold = 0.7;
    const cv::Scalar meanVal(104.0, 177.0, 123.0);

    bool do_detection = true;

    KCFTracker tracker;
    tracker.setParams(true, true, true, false);

    clock_t start_time, end_time;
    int frame_num;

    while (cap.isOpened())
    {
        cap.read(frame);

        if (frame.empty())
            break;
    
        start_time = clock();
        frame_num += 1;        

        if (do_detection)
        {
            #if defined(USE_SSD)
            cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);

            net_caffe.setInput(inputBlob, "data");
            cv::Mat detection = net_caffe.forward("detection_out");
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

            float confidence = detectionMat.at<float>(0, 2);

            if (confidence > 0.5)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(0, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(0, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(0, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(0, 6) * frameHeight);

                tracker.init(Rect(x1,y1,x2-x1,y2-y1), frame);
                cv::rectangle(frame, Point(x1,y1), Point(x2,y2), Scalar(0,255,0), 3);

                do_detection = false;

            }
            #elif defined(USE_YOLO)
            float* srcImg;
            cvtColor(frame, rgbImg, cv::COLOR_BGR2RGB);
            size_t srcSize=rgbImg.rows*rgbImg.cols*3*sizeof(float);
            srcImg=(float*)malloc(srcSize);

            imgConvert(rgbImg,srcImg);//将图像转为yolo形式
 
            float* resizeImg;
            size_t resizeSize=net->w*net->h*3*sizeof(float);
            resizeImg=(float*)malloc(resizeSize);
            imgResize(srcImg,resizeImg,frame.cols,frame.rows,net->w,net->h);//缩放图像
    
            network_predict(*net,resizeImg);//网络推理
            int nboxes=0;
            detection *dets=get_network_boxes(net,rgbImg.cols,rgbImg.rows,thresh,0.5,0,1,&nboxes,1);
    
            if(nms){
                do_nms_sort(dets,nboxes,classes,nms);
            }
    
            vector<cv::Rect>boxes;
            boxes.clear();
            vector<int>classNames;
    
            for (int i = 0; i < nboxes; i++){
                bool flag=0;
                int className;
                for(int j=0;j<classes;j++){
                    if(dets[i].prob[j]>thresh){
                        if(!flag){
                            flag=1;
                            className=j;
                        }
                    }
                }
                if(flag){
                    int left = (dets[i].bbox.x - dets[i].bbox.w / 2.)*frame.cols;
                    int right = (dets[i].bbox.x + dets[i].bbox.w / 2.)*frame.cols;
                    int top = (dets[i].bbox.y - dets[i].bbox.h / 2.)*frame.rows;
                    int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.)*frame.rows;
    
                    if (left < 0)
                        left = 0;
                    if (right > frame.cols - 1)
                        right = frame.cols - 1;
                    if (top < 0)
                        top = 0;
                    if (bot > frame.rows - 1)
                        bot = frame.rows - 1;
    
                    Rect box(left, top, fabs(left - right), fabs(top - bot));
                    tracker.init(box, frame);
                    cv::rectangle(frame, Point(left,top), Point(right,bot), Scalar(0,255,0), 3);

                    do_detection = false;
                    break;
                    boxes.push_back(box);
                    classNames.push_back(className);
                }
            }
            free_detections(dets, nboxes);
            free(srcImg);
            free(resizeImg);
            #endif
            
        }
        else
        {
            bool status;
            cv:: Rect trackedObj = tracker.update(frame, status);

            if (status)
                rectangle(frame, trackedObj, Scalar(0,255,0), 3);
            else
                do_detection = true;
        }
        end_time = clock();
        printf("frame number: %d, \n    algo time cost: %.2f ms.\n", frame_num, (end_time - start_time)*1000.0 / CLOCKS_PER_SEC);

        imshow("Frmae", frame);

        char key = waitKey(1);

        if (key == 'q')
            break;
    }
    #if defined(USE_YOLO)
    free_network(*net);
    #endif
    cap.release();

    return 0;
}
