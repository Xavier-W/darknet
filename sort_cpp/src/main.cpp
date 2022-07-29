#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "track.hpp"
#include "../../src/network.h"
#include "../../include/darknet.h"
#include "./improcess.h"

using namespace std;
using namespace cv;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
 
float get_color(int c, int x, int max){
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

int main(int argc, char **argv)
{
	TRACK tracker(30);
    VideoCapture cap("../../../../test.mp4");//读取视频，请自行修改相应路径
    string cfgfile = "../../cfg/yolov4.cfg";//读取模型文件，请自行修改相应路径
    string weightfile = "../../yolov4.weights";
    float thresh=0.5;//参数设置
    float nms=0.35;
    int classes=80;
 
    network *net=load_network((char*)cfgfile.c_str(),(char*)weightfile.c_str(),0);//加载网络模型
    set_batch_network(net, 1);
    
    Mat frame;
    Mat rgbImg;
 
    vector<string> classNamesVec;
    ifstream classNamesFile("../../cfg/coco.names");//标签文件coco有80类

 
    if (classNamesFile.is_open()){
        string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
 
    bool stop=false;
    clock_t start_time, end_time;

	int delay = 1;
	int frame_id = 0;

	while (1)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()){
			printf("fail to read.\n");
			break;
		}

		start_time = clock();
        cvtColor(frame, rgbImg, cv::COLOR_BGR2RGB);
 
        float* srcImg;
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

        vector<cv::Rect>det_boxes;
        det_boxes.clear();
        vector<int>classNames;
		vector<struct Bbox> bboxes;
		vector<BoundingBox> boxes;
		vector<TrackingBox> detFrameData;
        float confidence;

        for (int i = 0; i < nboxes; i++){
            bool flag=0;
            int className;
            for(int j=0;j<classes;j++){
                if(dets[i].prob[j]>thresh){
                    if(!flag){
                        flag=1;
                        className=j;
                        confidence = dets[i].prob[j];
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
                det_boxes.push_back(box);
                classNames.push_back(className);

                struct Bbox bbox;
				bbox.score = confidence;
				bbox.x1 = left;
				bbox.y1 = top;
				bbox.w = fabs(left - right);
				bbox.h = fabs(top - bot);
				bboxes.push_back(bbox);

            }
        }
        free_detections(dets, nboxes);
        end_time = clock();

		for (vector<struct Bbox>::iterator it = bboxes.begin(); it != bboxes.end(); it++)
		{
			boxes.push_back(BoundingBox(*it));
		}

		for (int i = 0; i < boxes.size(); ++i)
		{
			TrackingBox cur_box;
			cur_box.box = boxes[i].rect;
			cur_box.id = i;
			cur_box.frame = frame_id;
			detFrameData.push_back(cur_box);
		}
		++frame_id;

		vector<TrackingBox> tracking_results = tracker.update(detFrameData);

		for (TrackingBox it : tracking_results)
		{
			Rect object(it.box.x, it.box.y, it.box.width, it.box.height);
			rectangle(frame, object, tracker.randColor[it.id % 255], 2);
			putText(frame,
					to_string(it.id),
					Point2f(it.box.x, it.box.y),
					FONT_HERSHEY_PLAIN,
					2,
					tracker.randColor[it.id % 255]);
		}

        for(int i=0;i<det_boxes.size();i++){
            int offset = classNames[i]*123457 % 80;
            float red = 255*get_color(2,offset,80);
            float green = 255*get_color(1,offset,80);
            float blue = 255*get_color(0,offset,80);
 
            rectangle(frame,det_boxes[i],Scalar(0,0,0),1);
 
            String label = String(classNamesVec[classNames[i]]);
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            putText(frame, label, Point(det_boxes[i].x, det_boxes[i].y + labelSize.height),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0),1);
        }
		imshow("Webcam", frame);
		if ((waitKey(delay) == 113))
			break;
	}

	cap.release();
	destroyAllWindows();

	return 0;
}