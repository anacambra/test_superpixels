#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <set>

//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define MODE_LABEL_MEDIAN 1
#define MODE_LABEL_NOTZERO 2

class SuperPixel
{

    int _id;
    //mascara de la imagen original
    Mat _mask;
    int _numPixels;
    int _label;
    
    //neighbour ids
    set<int> _firstNeighbours;
    set<int> _secondNeighbours;
    
    MatND _labelHist;
    
    MatND hist_l;
    MatND hist_a;
    MatND hist_b;
    
/*    float l_median;
    float l_mean;
    float l_std;
    
    MatND hist_a;
    float a_median;
    float a_mean;
    float a_std;
    
    MatND hist_b;
    float b_median;
    float b_mean;
    float b_std;*/
    
    
    
   /* MatND hist_depth;
    //mediana sin ceros
    float d_median;
    //media
    float d_mean;
    //varianza
    float d_var;
    //precision
    float d_acc;
    float depth;*/
    
    
    public:
    
    SuperPixel(){ _id = -1; _numPixels = 0; _mask = Mat::zeros(0, 0, 0);  _label = -1; }
    ~SuperPixel(){} ;
    
    void initialize(int id,Mat mask, int label)
    {
        _id = id;
        _mask = mask.clone();
        _numPixels = countNonZero(_mask);
        _label = label;
    
    }
    int getId(){ return _id;}
    int getNumPixels(){ return _numPixels;}
    int getLabel(){ return _label;}
    Mat getMask(){ return _mask;}
    
    //first neighbours
    set<int> getFirstNeighbours(){ return _firstNeighbours;}
    void addFirstNeighbour(int n){ if (_id != n) _firstNeighbours.insert(n);}
    
    //second neighbours
    set<int> getSecondNeighbours(){ return _secondNeighbours;}
    void addSecondNeighbours(int n){ if (_id != n) _secondNeighbours.insert(n);}
    
    string toString(){ return "ID: " + to_string(_id) + " numPixels: " + to_string(_numPixels) + " label: " + to_string(_label);}
    
    //LABELS
    void setLabel(int l){_label = l;}
    
    float accLabel(int l){ return _labelHist.at<float>(l)/_numPixels;}
    
    int create_labelHist(Mat labels, int NUMLABELS, int mode = MODE_LABEL_MEDIAN)
    {
        int nbins = NUMLABELS; //  NUMLABELS levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(NUMLABELS) };
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        calcHist(&labels, 1, chnls, _mask, _labelHist,1,hsize,ranges);
        
       /*  printf("%d %d\n",_id,_numPixels);
        for( int h = 0; h < nbins; h++ )
        {
            float binVal = _labelHist.at<float>(h);
            printf("%d %f\n",h,binVal);
        }
         printf("-----\n");//*/
        
        double maxVal=0;
        Point maxBin;
        minMaxLoc(_labelHist, 0, &maxVal, 0, &maxBin);
        
       /* imshow("HIST labels",paintHistogram(_labelHist));
        waitKey(0);//*/
        
        switch (mode) {
                
            case MODE_LABEL_NOTZERO:
                if ( (maxBin.y == 0) && (maxVal < _numPixels)) //select other label before 0
                {
                    int b2,maxVal2=0.0;
                
                    for(int i=1;i < nbins;i++)
                    {
                        float value = _labelHist.at<float>(i);
                        if (value > maxVal2)
                        {
                            b2=i;
                            maxVal2=value;
                        }
                    }
                    _label = b2;
                }
                else
                    _label =  (int) maxBin.y;
                break;
                
            case MODE_LABEL_MEDIAN:
            default:
                //select the median bin
                float medianValue = (float)((_numPixels/2)+1)/(float)_numPixels;
                float count=0.0;
                int b;
                //for each bin
                for(b=0;b < nbins && count <= medianValue;b++)
            {
                float value = _labelHist.at<float>(b);
                count += (int) value;
            }
                _label = (b - 1);
                break;
        }
        
        return _label;

    }//create_labelHist

    /***************/
    //DESCRIPTORS
    /***************/
    
    Mat descriptorsLAB(Mat image, int NBINS_L = 101, int NBINS_AB=256)
    {
        // return Mat(1,101+256+256,CV_32FC1)
        Mat descriptor = Mat::zeros(1, NBINS_L+NBINS_AB+NBINS_AB, CV_32FC1);
        
        Mat image_out;
        cvtColor(image, image_out, CV_BGR2Lab);
        
        vector<Mat> spl;
        split(image_out,spl);
        
        //L <- L * 255/100 ; a <- a + 128 ; b <- b + 128
        
        //0 < L < 100
        int nbins = NBINS_L; // levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(100)};
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        spl[0]=spl[0] * (NBINS_L - 1)/255;
        
        calcHist(&spl[0], 1, chnls, _mask, hist_l,1,hsize,ranges);
        
        for(int b = 0; b < nbins; b++ )
        {
            float binVal = hist_l.at<float>(b);
            descriptor.at<float>(0,b)=binVal/(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        //-128 < a < 127
        int nbinsA = NBINS_AB; // levels
        int hsizeA[] = { nbinsA }; // just one dimension
        
        float rangeA[] = { 0, (const float)(256)};
        const float *rangesA[] = { rangeA };
        
        //spl[1]=spl[1] - 128;
        
        calcHist(&spl[1], 1, chnls, _mask, hist_a,1,hsizeA,rangesA);
        
        for(int b = 0; b < nbinsA; b++ )
        {
            float binVal = hist_a.at<float>(b);
            descriptor.at<float>(0,NBINS_L+b)= binVal /(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        //-128 < b < 127
        calcHist(&spl[2], 1, chnls, _mask, hist_b,1,hsizeA,rangesA);
        
        for(int b = 0; b < nbinsA; b++ )
        {
            float binVal = hist_b.at<float>(b);
            descriptor.at<float>(0,NBINS_L + NBINS_AB + b)= binVal /(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        return descriptor;
    }//descriptorsLAB
    
    Mat descriptorsRGB(Mat image, int BINS = 256)
    {
        // return Mat(1,256*3,CV_32FC1)
        Mat descriptor = Mat::zeros(1, BINS*3, CV_32FC1);
        
        vector<Mat> spl;
        split(image,spl);
        
        int nbins = BINS; // levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(256)};
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        MatND hist;
        
        //B
        calcHist(&spl[0], 1, chnls, _mask, hist,1,hsize,ranges);
        
        for(int b = 0; b < nbins; b++ )
        {
            float binVal = hist.at<float>(b);
            descriptor.at<float>(0,b)=binVal/(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        //G
        calcHist(&spl[1], 1, chnls, _mask, hist,1,hsize,ranges);
        
        for(int b = 0; b < nbins; b++ )
        {
            float binVal = hist.at<float>(b);
            descriptor.at<float>(0,BINS+b)=binVal/(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        calcHist(&spl[2], 1, chnls, _mask, hist,1,hsize,ranges);
        
        for(int b = 0; b < nbins; b++ )
        {
            float binVal = hist.at<float>(b);
            descriptor.at<float>(0,BINS+BINS+b)=binVal/(float)_numPixels;
            //printf("%d %f\n",b,binVal);
        }
        
        return descriptor;
    }//descriptorsRGB
    
    Mat descriptorsPEAKS(Mat image, int BINS = 256)
    {
        Mat descriptor = Mat::zeros(1, BINS, CV_32FC1);
        
        Mat img;
        cvtColor(image, img, CV_BGR2GRAY);
        
        int nbins = BINS; // levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(256)};
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        MatND hist;
        
        calcHist(&img, 1, chnls, _mask, hist,1,hsize,ranges);
        
        for(int b = 0; b < nbins; b++ )
        {
            float binVal = hist.at<float>(b);
            descriptor.at<float>(0,b)=(binVal / (float)_numPixels);
            //printf("%d %f\n",b,binVal);
        }
        
       // imshow("GRAY hist",paintHistogram(hist));//waitKey(0);
        return descriptor;
    }//descriptorsPEAKS
    
    
    
    //VECINOS
    
    
    /***************************************/
    
    Mat paintHistogram (MatND hist)
    {
        
       // MatND hist = _labelHist.clone();
        
        // Plot the histogram

        int hist_w = 300; int hist_h = 200;
        int bin_w = cvRound( (double) hist_w/hist.size[0] );
        
        Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
        normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        
        for( int i = 1; i < hist.size[0]; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                 Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                 Scalar( 255, 0, 0), 2, 8, 0  );
        }
        
        return histImage;
    }//paintHistogram
    

};

#endif // SUPERPIXEL_H
