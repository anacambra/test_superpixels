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


class labelSet
{
    //pascal
    const string pascal[21]= {"background","aeroplane", "bicycle", "bird", "boat","bottle","bus","car","cat","chair","cow","table","dog","horse",
        "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};
    //pascal-context
    const string pascalcontext[60]= { "background ","aeroplane", "bicycle", "bird", "boat","bottle","bus","car","cat","chair","cow","table","dog","horse",
        "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
        "bag","bed","bench","book","building","cabinet","ceiling","cloth",
        "computer","cup","door","fence","floor","flower","food","grass","ground",
        "keyboard","light","    mountain","mouse","curtain","platform","sign",
        "plate","road","rock","shelves","sidewalk","sky","snow","bedclothes",
        "track","tree","truck","wall","water","window","wood"};
    
    //text
    const string text[2]= { "background ","text"};
    
    string *_labels;
    
public:
    labelSet(int NUMLABELS)
    {
        _labels = new string[NUMLABELS];
        
        if (NUMLABELS == 60)
            copy(pascalcontext, pascalcontext+NUMLABELS, _labels);
        else if (NUMLABELS == 21)
            copy(pascal, pascal+NUMLABELS, _labels);
        else if (NUMLABELS == 2)
            copy(text, text+2,_labels);
    };
    
    ~labelSet(){};
    
    //paint random labels (0..numLabels)
    Mat paintLabelRandom(Mat label, int numLabels,Mat *leyend)
    {
        int t=0,f=0;
        Mat bgr = Mat::zeros(label.rows, label.cols, CV_8UC3);
        RNG rng(0);
        Scalar color;
        
        cout << "----------------------------------------\n";
        
        for (int i=0; i<  numLabels; i++)
        {
            Mat mask = (label == i);
            
            if (i == 0)
                color = Scalar (0,0,0);
            else
            {
                color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                bgr.setTo(color,mask);
                
                if (countNonZero(mask) > 0)
                {
                    std::cout << "label " << i << " "<< _labels[i] <<
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
            }//if (i==0)
        }//for i
        
        std::cout << "----------------------------------------\n";
        
        return bgr;
    }

    
};
