#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <set>

//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

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
        _numPixels = countNonZero(mask);
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
    
    string toString(){ return "ID: " + to_string(_id) + " numPixels: " +to_string(_numPixels) + " label: " + to_string(_label);}
    
    //LABELS
    void setLabel(int l){_label = l;}
    
    int create_labelHist(Mat labels, int NUMLABELS)
    {
        int nbins = NUMLABELS; //  NUMLABELS levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(NUMLABELS - 1) };
        const float *ranges[] = { range };
        int chnls[] = {0};
        calcHist(&labels, 1, chnls, _mask, _labelHist,1,hsize,ranges);
        
        for( int h = 0; h < nbins; h++ )
        {
            float binVal = _labelHist.at<float>(h);
            printf("%d %f\n",h,binVal);
        }
        
        /*double maxVal=0;
        Point maxBin;
        minMaxLoc(_labelHist, 0, &maxVal, 0, &maxBin);
        
        printf("MAX: %f %d %d\n",maxVal,maxBin.x, maxBin.y);*/
        //MODA: bin = maxBin.y
        
        
        //select the median bin
        int medianValue = (_numPixels/2)+1;
        int count=0;
        int b;
        //for each bin
        for(b=0;b < nbins && count <= medianValue;b++)
        {
            float value = _labelHist.at<float>(b);
            count += (int) value;
        }
        
        _label = (b - 1);
        
        
        return (b-1);
        
        
    }//create_labelHist

    /***************/
    //DESCRIPTORS
    /***************/
    
    void descriptors(Mat image, Mat *dvec)
    {
        //show
        Mat image_out;
        cvtColor(image, image_out, CV_BGR2Lab);
        
        vector<Mat> spl;

        split(image_out,spl);
        
       // double max, min;
        //L <- L * 255/100 ; a <- a + 128 ; b <- b + 128
        
       /* minMaxLoc(spl[0], &min, &max, 0, 0);
        printf("%f L %f\n",min* 100/255,max* 100/255);
        minMaxLoc(spl[1], &min, &max, 0, 0);
        printf("%f a %f\n",min-127,max-127);
        minMaxLoc(spl[2], &min, &max, 0, 0);
        printf("%f b %f\n",min-127,max-127);*/
        
        //0 < L < 100
        int nbins = 100; // levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(nbins - 1) };
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        spl[0]=spl[0] * 100/255;
        
        calcHist(&spl[0], 1, chnls, _mask, hist_l,1,hsize,ranges);
        
        for(int h = 0; h < nbins; h++ )
        {
            float binVal = hist_l.at<float>(h);           
            dvec->at<float>(0,h)=binVal;
           // printf("%f ",binVal);
        }
        
       // spl[1]=spl[1] - 128;
    /*    calcHist(&spl[1], 1, chnls, _mask, hist_a,1,hsize,ranges);
        
        for( int h = 0; h < nbins; h++ )
        {
            float binVal = hist_a.at<float>(h);
            printf("%f ",binVal);
        }
        
        //spl[2]=spl[2] - 128;
        
        calcHist(&spl[2], 1, chnls, _mask, hist_b,1,hsize,ranges);
        
        for( int h = 0; h < nbins; h++ )
        {
            float binVal = hist_b.at<float>(h);
            printf("%f ",binVal);
        }
        */
        Mat out;
        cvtColor(_mask,out,CV_GRAY2BGR);
        bitwise_and(image, out, out);
        imshow("out",out);//l
       // waitKey(0);//*/
        
    
        
    }
    //meanColor RGB
    //histogram color RGB, LAB, edges
    
    
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
