//
//  labelSet.cpp
//  test_sup
//
//  Created by Ana Cambra on 17/12/15.
//
//

#include <stdio.h>
#include <string>

#include <iostream>
//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define SEGNET 12
#define TEXT   2
#define PASCAL 21
#define PASCALCONTEXT 60

class labelSet
{

    vector<string> pascal= {"background","aeroplane", "bicycle", "bird", "boat","bottle","bus","car","cat","chair",
                            "cow","table","dog","horse", "motorbike","person","pottedplant","sheep","sofa","train",
        "tvmonitor"};

    vector<string> pascalcontext = { "background ","aeroplane", "bicycle", "bird", "boat","bottle","bus","car","cat","chair","cow","table","dog","horse",
        "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
        "bag","bed","bench","book","building","cabinet","ceiling","cloth",
        "computer","cup","door","fence","floor","flower","food","grass","ground",
        "keyboard","light","mountain","mouse","curtain","platform","sign",
        "plate","road","rock","shelves","sidewalk","sky","snow","bedclothes",
        "track","tree","truck","wall","water","window","wood"};
    
    
    vector<string> pascalcontextNoObjects = {"background","building","cabinet","ceiling","door",
                                              "fence","floor","grass","ground","light",
                                              "mountain","platform","sign","road","shelves",
                                              "sidewalk","sky","track","tree","wall",
                                              "window","wood","objects"};
    
    vector<string> pascalcontextNoObjects2 = {"background","building","cabinet","ceiling","door",
        "fence","floor","grass","ground","light",
        "mountain","platform","sign","road","shelves",
        "sidewalk","sky","track","tree","wall",
        "window","wood"};
    
    vector<string> text = { "background ","text"};
    
    vector<string> segnet= {"sky","building", "pole", "roadMarking", "road","pavement",
                            "tree","sign","fence","vehicle","pedestrian","bike"};
    vector<string> segnetNoObjects= {"sky","building", "pole", "roadMarking", "road","pavement",
        "tree","sign","fence","vehicle","pedestrian","bike"};
    //roadMarking == road == pavement
    //vehicle == pedestrian == bike
    
    
    vector<string> _labels;
    
public:
    unsigned char _DEBUG = 1;
    
    labelSet(int NUMLABELS)
    {
        if (NUMLABELS == PASCALCONTEXT)
            _labels = pascalcontext;
        else if (NUMLABELS == PASCAL)
            _labels = pascal;
        else if (NUMLABELS == 23)
            _labels = pascalcontextNoObjects;
        else if (NUMLABELS == 22)
            _labels = pascalcontextNoObjects2;
        else if (NUMLABELS == TEXT)
            _labels = text;
        else if (NUMLABELS == SEGNET)
            _labels = segnet;
        
    };
    
    ~labelSet(){ // delete[] _labels;
        _labels.clear();};
    
    string getLabel(int i){return _labels[i];}
    
    //paint random labels (0..numLabels)
    Mat paintLabelRandom(Mat label, int numLabels,Mat *leyend)
    {
        int t=0,f=0;
        Mat bgr = Mat::zeros(label.rows, label.cols, CV_8UC3);
        RNG rng(0);
        Scalar color;
        
        if (_DEBUG == 1) cout << "----------------------------------------\n";
        
        for (int i=0; i<  numLabels; i++)
        {
            Mat mask = (label == i);
            
            if ((i == 0) && (numLabels != SEGNET))
                color = Scalar (0,0,0);
            else
            {
                if (numLabels == SEGNET)
                    color = colorSegnet(i);
                else
                    color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                
                bgr.setTo(color,mask);
            
                if (countNonZero(mask) > 0)
                {
                    if (_DEBUG == 1)std::cout << "label " << i << " "<< _labels[i] <<
                    " RGB: (" << color[2] << "," << color[1]<< "," << color[0] << ")" << "\n";
                    rectangle(*leyend, Point(((label.cols/4)*f),(20*t)), Point(((label.cols/4)*f)+30,20+(20*t)), color,-1);
                    putText(*leyend, _labels[i], Point(((label.cols/4)*f)+30, 20+(20*t)), FONT_HERSHEY_COMPLEX, 0.5, color, 1.0);
                    
                    t= t+1;
                    if (t==5)
                    {
                        t=0;
                        f=f+1;
                    }
                }//if (countNonZero(mask) > 0)
                mask.release();
           }//if (i==0)
        }//for i
        
        if (_DEBUG == 1) std::cout << "----------------------------------------\n";
        
        return bgr.clone();
    }
    
    Scalar colorSegnet(int label)
    {
        if (label == 0)
            return Scalar(128,128,128);
        
        if (label == 1)
            return Scalar(0,0,128);
        
        if (label == 2)
            return Scalar(128,192,192);
        
        if (label == 3 )
            return Scalar(0,69,255);
        
        if (label == 4)
            return Scalar(128,64,128);
        
        if (label == 5)
            return Scalar(222,40,60);
        
        if (label == 6)
            return Scalar(0,128,128);
        
        if (label == 7)
            return Scalar(128,128,192);
        
        if (label == 8)
            return Scalar(128,64,64);
        
        if (label == 9)
            return Scalar(128,0,64);
        
        if (label == 10)
            return Scalar(0,64,64);
        
        if (label == 11)
            return Scalar(192,128,0);
        
        return Scalar(0,0,0);

    }
    
    Mat convert2pascalcontextNoObjects(Mat pascal)
    {
        Mat newPascal = Mat::zeros(pascal.rows, pascal.cols, CV_8UC1);
        
        for(int i=0; i < pascalcontext.size(); i++)
        {
            Mat mask = (pascal == i);
            if (countNonZero(mask) > 0)
            {
                bool found = false;
                for (int j=0; j < pascalcontextNoObjects.size(); j++)
                {
                    if (pascalcontext[i].compare(pascalcontextNoObjects[j]) == 0)
                    {
                        printf("----> %d %s - %d %s \n",i,pascalcontext[i].c_str(),j,pascalcontextNoObjects[j].c_str());
                        newPascal.setTo(j,mask);//= j;
                        //imshow("new",newPascal == j);waitKey(0);
                        //getchar();
                        found =true;
                        break;
                    }
                }
                
                if (!found) newPascal.setTo(((int)pascalcontextNoObjects.size()-1),mask);
            }
        }
        return newPascal;
    }
    
};
