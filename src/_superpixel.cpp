#include "superpixel.h"

#include <stdio.h>

#define showInfo 0

/**
 * Constructor
 * @param path Path completo de la imagen utilizada como base
 */

SuperPixel::SuperPixel()
{
    
    id = -1;
}

/**
 * Destructor
 */
SuperPixel::~SuperPixel() {

}

void SuperPixel::init(int id,Mat mask,int numPixels)
{
    
    this->id = id;
    this->mask = mask.clone();
    this->numPixels=numPixels;
    //imshow("mask",this->mask);
}

float SuperPixel::medianHistogram(MatND hist,int num)
{
    int numValue= (num/2)+1;
    int count=0;
    //for each bin
    for(int b=0;b<hist.rows;b++) {
        
        float value = hist.at<float>(b);
        count += (int) value;
        if (count >=numValue)
        {
            return (float)b/255.0f;
        }
    }
    return 0.0f;
}



float SuperPixel::meanHistogram(MatND hist,int num)
{
 
    float m=0;
    for(int b=0;b<hist.rows;b++) {
        m = m + (hist.at<float>(b)*(b/255.0f));
    }
    return (m/float(num));
}

float SuperPixel::varHistogram(MatND hist,float mean,int num)
{
    float m=0;
    for(int b=0;b<hist.rows;b++) {
        float value = mean - (b/255.0f);
        if (hist.at<float>(b) != 0)
        {
            value = value * value;
            value = value * hist.at<float>(b);
            m = m + value;
        }
        
       // m = m + ((mean - (hist.at<float>(b)*(b/255.0f)))*(mean - (hist.at<float>(b)*(b/255.0f))));
    }
    return (m/float(num));
}


void SuperPixel::meanVarDepth(Mat depth)
{
    
    //Mat notZero;
    //notZero = (depth != 0.0f);
    //combinar las 2 mascaras
    //bitwise_and(mask,notZero,notZero)
    
    Scalar m,std;
    meanStdDev(depth, m, std);//,mask);//notZero);
    d_mean = (float)m.val[0];
    printf("id = %d media %0.2f \n",id,d_mean);
    d_var = (float)std.val[0]*(float)std.val[0];

}

void SuperPixel::meanVarDepthWithoutCeros(Mat depth)
{
    
    Mat notZero;
    notZero = (depth != 0.0f);
    //combinar las 2 mascaras
    bitwise_and(mask,notZero,notZero);
    
    Scalar m,std;
    meanStdDev(depth, m, std,notZero);
    d_mean = (float)m.val[0];
    printf("id = %d media %0.2f \n",id,d_mean);
    d_var = (float)std.val[0]*(float)std.val[0];
    
}


float SuperPixel::medianDepth(Mat _depth)//;,MatND h)//, float accurancy)
{
    //Histograma sin depth cero
    
    Mat notZero;
    //compare(_depth,0,notZero,CMP_GT);
    notZero = (_depth != (float)0.0f);
    //combinar las 2 mascaras
    
    bitwise_and(mask,notZero,notZero);
    int withDepth = countNonZero(notZero);
    
    if (withDepth > 0)
    {
        
        //histograma de la depth
        int nbins = 256; // lets hold 256 levels
        int hsize[] = { nbins }; // just one dimension
        float range[] = { 0, 1 };
        const float *ranges[] = { range };
        int chnls[] = {0};
        MatND d_hist;
        calcHist(&_depth, 1, chnls, notZero, d_hist,1,hsize,ranges);
        
        this->hist_depth = d_hist.clone();
        
       // d_mean = meanHistogram(hist_depth, withDepth);//m.val[0];
       // d_var = varHistogram(hist_depth, d_mean, withDepth); //std.val[0];
        
        this->d_median=medianHistogram(hist_depth,withDepth);
        this->d_acc=(float)withDepth / (float)numPixels;
        //
     /*   for(int b=0;b<hist_depth.rows;b++) {
            
            int value = (int)hist_depth.at<float>(b);
            printf("%d %d \n",b,value);
           
        
        }*/
        
        
         
    }
    else
    {
        this->d_median=0;
        this->d_acc=0;
       // d_mean = 0;
       // d_var = 0;
    }
    
     //  showHistogram(h);
   if (showInfo == 1)
       printf("*DEPTH id=%4d\tmedian= %f\taccurancy= %f (%d/%d)\tmean= %f\tVar= %f\n",id,d_median,d_acc,
           withDepth,countNonZero(mask),d_mean,sqrt(d_var));
    return  this->d_median;

}

void SuperPixel::descriptorsLab(Mat image){

    //image en formato lab
    //separar le histograma de cada canal
    std::vector<cv::Mat> colorPlanes;
    cv::split(image, colorPlanes);
    
    /* printf("_lab: %d %d %d %f %f %f \n",
     _lab.at<Vec3b>(0,0)[0],
     _lab.at<Vec3b>(0,0)[1],
     _lab.at<Vec3b>(0,0)[2],
     (float)_lab.at<Vec3b>(0,0)[0]*100.0/255.0,
     (float) _lab.at<Vec3b>(0,0)[1]-128.0,
     (float)_lab.at<Vec3b>(0,0)[2]-128.0);*/
    
    //pintar histograma
    
    /// Establish the number of bins
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    
    bool uniform = true; bool accumulate = false;

    
    /// Compute the histograms:
    calcHist( &colorPlanes[0], 1, 0, mask, hist_l, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &colorPlanes[1], 1, 0, mask, hist_a, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &colorPlanes[2], 1, 0, mask, hist_b, 1, &histSize, &histRange, uniform, accumulate );
    
    //calcular media, mediana y desvicion típica
    Scalar m,std;
    meanStdDev(hist_l, m, std);
    l_median = medianHistogram(hist_l, numPixels);
    l_mean = float(m.val[0]);
    l_std = float(std.val[0]);
    
    meanStdDev(hist_a, m, std);
    a_median = medianHistogram(hist_a, numPixels);
    a_mean=float(m.val[0]);
    a_std=float(std.val[0]);
    
    meanStdDev(hist_b, m, std);
    b_median = medianHistogram(hist_b, numPixels);
    b_mean = float(m.val[0]);
    b_std=float(std.val[0]);
    
    if (showInfo == 1)
    {
    
        printf("*L id=%4d\tmedian= %f\tmean= %f\tStdDev= %f\n",id,l_median,l_mean,l_std);
        printf("*a id=%4d\tmedian= %f\tmean= %f\tStdDev= %f\n",id,a_median,a_mean,a_std);
        printf("*b id=%4d\tmedian= %f\tmean= %f\tStdDev= %f\n",id,b_median,b_mean,b_std);
    }
    
   // normalize(hist_l,1.0);
  /*  normalize(hist_l, hist_l, 1.0,0.0, NORM_L1, -1, Mat() );
    normalize(hist_a, hist_a,1.0,0.0, NORM_L1, -1, Mat() );
    normalize(hist_b, hist_b, 1.0,0.0, NORM_L1, -1, Mat() );*/
    
    
    // Draw the histograms for B, G and R
  /*  int hist_w = 256*2; int hist_h = 300;
    
    Mat l( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat a( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat b( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    
    /// Normalize the result to [ 0, 1 ]
    normalize(l_hist, l_hist, 0.0,1.0, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist,0.0,1.0, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0.0,1.0, NORM_MINMAX, -1, Mat() );
    
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        int intensity =  (int)(l_hist.at<float>(i)*(float)hist_h);
        
        line(l,cvPoint(32+2*i,hist_h),cvPoint(32+2*i,hist_h-intensity), CV_RGB(250,250,250), 1, 8, 0);
        
        intensity =  (int)(g_hist.at<float>(i)*(float)hist_h);
        line(a,cvPoint(32+2*i,hist_h),cvPoint(32+2*i,hist_h-intensity), CV_RGB(250,250,250), 1, 8, 0);
        
        intensity =  (int)(r_hist.at<float>(i)*(float)hist_h);
        line(b,cvPoint(32+2*i,hist_h),cvPoint(32+2*i,hist_h-intensity), CV_RGB(250,250,250), 1, 8, 0);
        printf("\n %f",l_hist.at<float>(i));
        
    }
   imshow("l", l );
    imshow("a", a );
     imshow("b", b );
   */

}

float SuperPixel::cmpHistogram(MatND a,MatND b,int mode)
{
    //0 perfect
    //1 mismatch
    float cost;
    
    //es posible q un superpixel no tenga histograma then devolver 1
    if ((a.rows != b.rows) || (a.cols != b.cols) || a.rows == 0 || b.rows == 0 )
        return 1.0f;
    
    
    switch (mode) {
        case CV_COMP_CORREL:
            //-1 mismatch
            //1 Perfect
            
            // -1 __ 0 __ 1  x1 __ x __ x2
            //  1 __ y __ 0  y1 __ y __ y2
            
            // y = (((x-x1)/(x2-x1))* (y2-y1))+ y1
            
            cost = float(compareHist(a,b, CV_COMP_CORREL));
            //cost = (((cost-(-1))/(1-(-1)))*(0-1))+(1);
            
            break;
            
        case CV_COMP_CHISQR:
            //TODO normalizar
            //0 perfect
            // mismatch unbound ????
            cost = float(compareHist(a,b, CV_COMP_CHISQR));
            break;
            
        case CV_COMP_INTERSECT:
            //1 Perfect
            //0 mismatch
            //TODO normalizar
            
            cost =  float((compareHist(a,b, CV_COMP_INTERSECT)));
            break;
            
            
        case CV_COMP_BHATTACHARYYA | CV_COMP_HELLINGER:
            //0 perfect
            //1 mismatch
            cost =  float(compareHist(a,b, CV_COMP_BHATTACHARYYA));
            break;
    }
    return cost;

}

void SuperPixel::showDepth(){
    showHistogram(hist_depth);
}

void SuperPixel::showHistogram(MatND hist){

    //mostrar histograma
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    // int rows = 64; //default height size
    int cols = 256;//hist.rows; //get the width size from the histogram
    Mat histImg = Mat::zeros(256,256*2,  CV_8UC3);
     
    for(int b=0;b<cols;b++) {
        float value = hist.at<float>(b);
        int normalized = cvRound(value * 300 / maxVal);
        line(histImg,cvPoint(b*2, 256), cvPoint(b*2, 256 - normalized), CV_RGB(255, 255, 255));
        printf("%f\n",(float)value);fflush(stdout);
    };
    
    imshow("histograma " , histImg);
    
}
