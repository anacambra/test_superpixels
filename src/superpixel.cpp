#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <set>

//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "utilsCaffe.cpp"

#include "labelSet.cpp"

using namespace cv;
using namespace std;

#define MODE_LABEL_MEDIAN 1
#define MODE_LABEL_NOTZERO 2

class SuperPixel
{
    int _id;
    //mask original image
    Mat _mask;
    int _numPixels;
    int _label;
    
    //neighbour ids
    set<int> _firstNeighbours;
    set<int> _secondNeighbours;
    
    MatND _labelHist;
    MatND _labelSegmentation;
    int maxLabelSegmentation;
    MatND _labelFirstSegmentation;
    
    MatND hist_l;
    MatND hist_a;
    MatND hist_b;
    
    
    unsigned char _DEBUG = 0;
    
public:
    
    float timeLAB = 0.0;
    
    SuperPixel(){ _id = -1; _numPixels = 0; _mask = Mat::zeros(0, 0, 0);  _label = -1; }
    ~SuperPixel(){ /*if (_mask.rows!=0 && _mask.cols !=0) _mask.release();*/}
    
    void activeDEBUG(){ _DEBUG = 1;}
    void desactiveDEBUG(){ _DEBUG = 0;}
    
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
    
    MatND getLabelSegmentation(){ return _labelSegmentation;}
    MatND getFirstLabelSegmentation(){ return _labelFirstSegmentation;}
    
    void normalizeLabelFirstSegmentation(float i)
    {
        _labelFirstSegmentation = _labelFirstSegmentation / (float) i;
       // labelSet val(60);
       // printf("\nFIRST Neigbourd segmentation: %d %s\n",maxLabelSegmentation,val.getLabel(maxLabelSegmentation).c_str());
        for( int h = 0; h < 23; h++ )
        {
            float binVal = _labelFirstSegmentation.at<float>(h);
           /* if (binVal != 0)
            {
               // printf("%d %f\n",h,binVal);
                //printf("LABEL: %d %s  %f\n",h,val.getLabel(h).c_str(),binVal);
            }*/
        }
        //getchar();
    }
    
    int addHistogramLabelSegmentation(MatND hist)
    {
       //return label maxima
       /* printf("NEW HIST\n");
        for( int h = 0; h < 60; h++ )
        {
            float binVal = hist.at<float>(h);
            if (binVal != 0)printf("%d %f\n",h,binVal);
        }
        printf("-----\n");//*/
        

        if (_labelFirstSegmentation.rows == 0)
            _labelFirstSegmentation = hist.clone();
        
        else{//*/
           /* printf(" ACTUAL\n");
            for( int h = 0; h < 60; h++ )
            {
                float binVal = _labelFirstSegmentation.at<float>(h);
                if (binVal != 0)printf("%d %f\n",h,binVal);
            }
           // printf("-----\n");//*/
        
        /*printf("NEW HIST\n");
        for( int h = 0; h < 60; h++ )
        {
            float binVal = hist.at<float>(h);
            if (binVal != 0)printf("%d %f\n",h,binVal);
        }*/
       
        
            MatND dest;
            add(hist,_labelFirstSegmentation,_labelFirstSegmentation);

                        
            //bitwise_or(_labelNeighbours,hist,_labelNeighbours);
           // _labelNeighbours = _labelNeighbours + hist;
        }
        
        
        /*for( int h = 0; h < 60; h++ )
        {
            float binVal = _labelFirstSegmentation.at<float>(h);
            //if (binVal != 0)printf("label %s %d %f\n",labels.getLabel(h).c_str(),h,binVal);
        }//*/
        
       // normalize(_labelSegmentation,_labelSegmentation);
        
        //max bin in histogram
        double maxVal=0;
        Point bin;
        minMaxLoc(_labelFirstSegmentation, 0, &maxVal, 0, &bin);

       /* printf("MAX: %d %f ", bin.y,maxVal);
        printf("-----\n");//getchar();*/
        
        return (int)bin.y;
        
    }
    
    //first neighbours
    set<int> getFirstNeighbours(){ return _firstNeighbours;}
    void addFirstNeighbour(int n){ if (_id != n) _firstNeighbours.insert(n);}
    
    //second neighbours
    set<int> getSecondNeighbours(){ return _secondNeighbours;}
    void addSecondNeighbours(int n){ if (_id != n) _secondNeighbours.insert(n);}
    
    //string toString(){ return "ID: " + _id + " numPixels: " + to_string(_numPixels) + " label: " + to_string(_label);}
    
    //LABELS
    void setLabel(int l){_label = l;}
    
    float accLabel(int l)
    {
        if (_numPixels != 0 && _labelHist.rows > 0 && _labelHist.cols > 0)
            return _labelHist.at<float>(l)/_numPixels;
        else return -1;
    
    }
    
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
    
    int create_labelSegmentation(Mat seg, int NUMLABELS, int mode = MODE_LABEL_NOTZERO)
    {
        
        int nbins = NUMLABELS; //  NUMLABELS levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(NUMLABELS) };
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        //resize
        
        calcHist(&seg, 1, chnls, _mask, _labelSegmentation,1,hsize,ranges);
        
        _labelSegmentation = _labelSegmentation / _numPixels;
        
        double maxVal=0;
        Point maxBin;
        minMaxLoc(_labelSegmentation, 0, &maxVal, 0, &maxBin);
        
        /* imshow("HIST labels",paintHistogram(_labelHist));
         waitKey(0);//*/
        
        int label;
        switch (mode) {
                
            case MODE_LABEL_NOTZERO:
                if ( (maxBin.y == 0) && (maxVal < _numPixels)) //select other label before 0
                {
                    int b2,maxVal2=0.0;
                    
                    for(int i=1;i < nbins;i++)
                    {
                        float value = _labelSegmentation.at<float>(i);
                        if (value > maxVal2)
                        {
                            b2=i;
                            maxVal2=value;
                        }
                    }
                    label = b2;
                }
                else
                    label =  (int) maxBin.y;
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
                    float value = _labelSegmentation.at<float>(b);
                    count += (int) value;
                }
                label = (b - 1);
                break;
        }
        
        maxLabelSegmentation = label;
        return label;
        
    }//create_labelHist
    
    /***************/
    //DESCRIPTORS
    /***************/
    
    Mat descriptorsLAB(Mat image, int NBINS_L = 101, int NBINS_AB=256)
    {
        clock_t start = clock();
        
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
        
        spl[0].release();
        spl[1].release();
        spl[2].release();
        image_out.release();
        
        timeLAB = (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
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
        
        spl[0].release();
        spl[1].release();
        spl[2].release();
        
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
        
        if (_DEBUG == 1) imshow("Hist GRAY",paintHistogram(hist));
        hist.release();
        img.release();
        return descriptor;
    }//descriptorsPEAKS
    
    Mat descriptorsEDGES(Mat image, int BINS = 4, int mode = 1 )
    {
        
        //mode = 0: histogram
        //mode = 1: moments mean stdDes skew kurtosis
        //mode = 2: hist + moments
        
        int momentsSIZE = 4;
        //image is BGR float
        Mat out;
        
        
        Mat descriptor = Mat::zeros(1, BINS, CV_32FC1);
        cvtColor(image, out, CV_BGR2GRAY);
        out.convertTo(out, CV_32FC1,1.0/255.0,0);
        
        if (out.rows != _mask.rows || out.cols != _mask.cols)
        {
            resize(out, out, Size(_mask.cols,_mask.rows));
        }

        
        //HISTOGRAM
        if (mode == 0 || mode == 2)
        {
            int nbins = BINS; // levels
            int hsize[] = { nbins }; // just one dimension
            
            float range[] = { 0, (const float)(1.0)};
            const float *ranges[] = { range };
            int chnls[] = {0};
            
            MatND hist;
            
            calcHist(&out, 1, chnls, _mask, hist,1,hsize,ranges);
            
            for(int b = 0; b < nbins; b++ )
            {
                float binVal = hist.at<float>(b);
                if (mode == 0 || mode == 2) descriptor.at<float>(0,b)=(binVal / (float)_numPixels);
                // printf("%d \t %f\n",b,binVal);
            }
            
            if (_DEBUG == 1) imshow("Hist EDGES",paintHistogram(hist));
            
            hist.release();
        }
        
        //MOMENTS
        if (mode == 1 || mode == 2)
        {
           /* cvtColor(image, out, CV_BGR2GRAY);
            out.convertTo(out, CV_32FC1,1.0/255.0,0);*/
            
            
            Mat mask = Mat::zeros(out.rows,out.cols,CV_32FC1);
            out.copyTo(mask, _mask);
            
            double min, max;
            minMaxLoc(mask, &min, &max);
            Scalar     mean, stddev;
            meanStdDev ( mask, mean, stddev, _mask );
            
            float mc3,mc4;

            mc3=0;
            mc4=0;

            for (int x = 0; x < mask.rows; x++)
            {
                for (int y = 0; y < mask.cols; y++)
                {
                    if (_mask.at<uchar>(x,y) != 0)
                    {
                        mc3 = mc3 + ( pow( (float)mask.at<float>(x,y) - (float)mean.val[0] , 3) );
                        mc4 = mc4 + ( pow( (float)mask.at<float>(x,y) - (float)mean.val[0] , 4) );
                    }

                }
            }
            mc3 = mc3  / (float) _numPixels;
            mc4 = mc4  / (float) _numPixels;
            
            float s = 0.0;
            float k = 0.0;
            //normalize
            if ((float)stddev.val[0] != 0)
            {
                 s = (mc3 / pow((float)stddev.val[0],3) ) / (float) _numPixels;
                 k = (mc4 / pow((float)stddev.val[0],4))/ (float) _numPixels ;
            }

            
           if (_DEBUG == 1) printf("Superpixel::EDGES %d [%f,%f] mean %f stdDev %f s %f k %f \n",_id, mc3,mc4, mean.val[0],stddev.val[0],s,k); //getchar();//*/

            int b;
            if (mode == 1)
                b = 0;
            else
                b = BINS - momentsSIZE;
            
            descriptor.at<float>(0,b++) = mean.val[0];
            descriptor.at<float>(0,b++) = stddev.val[0];
            descriptor.at<float>(0,b++) =  s;
            descriptor.at<float>(0,b++) =  k;
           
            mask.release();
            
            return  descriptor;
        }

        out.release();

        return  descriptor;
        
    }//descriptorsEDGES
    
    /*Mat descriptorsCAFFE(Mat image, string CAFFE_LAYER = "fc7", int NUMCAFFE = 4096)
    {
         utilsCaffe caffe("/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel","/Users/acambra/Dropbox/test_caffe/deploy.prototxt");
         //crop!!!!!!
         //Mat cv_img = imread("/Users/acambra/Dropbox/test_caffe/cat.jpg", CV_LOAD_IMAGE_COLOR);; // Input
         //caffe.features(cv_img, "conv3");
        return caffe.features(image, "fc7").clone();
    }//descriptorsCAFFE*/
    
    Mat descriptorsEDGESDIR(Mat edges,Mat edgesDIR, int NBINS_EDGESDIR = 8)
    {
        Mat descriptor = Mat::zeros(1, NBINS_EDGESDIR, CV_32FC1);
        
        float maxProb = 0.0;
        
        for( int h = 0; h < NBINS_EDGESDIR; h++ )
            descriptor.at<float>(0,h)=0.0;
        
       
        //edges [0-1] probabiltities Max= 1.0*_numPixels
        cvtColor(edges, edges, CV_BGR2GRAY);
        edges.convertTo(edges, CV_32FC1,1.0/255.0,0);
        
        if (edges.rows != _mask.rows || edges.cols != _mask.cols)
        {
            resize(edges, edges, Size(_mask.cols,_mask.rows));
        }
        if (edgesDIR.rows != _mask.rows || edgesDIR.cols != _mask.cols)
        {
            resize(edgesDIR, edgesDIR, Size(_mask.cols,_mask.rows));
        }
        
        
        //edgesDir [0-3.14]
        double min, max;
        minMaxLoc(edgesDIR, &min, &max);
        

        for (int i=0; i< edgesDIR.rows; i++)
        {
            for (int j=0; j< edgesDIR.cols; j++)
            {
                if (_mask.at<uchar>(i,j) != 0)
                {
                    int b = (int)(edgesDIR.at<float>(i,j)/(max/(float)NBINS_EDGESDIR));
                    descriptor.at<float>(0,b) += edges.at<float>(i,j);
                    
                    maxProb += edges.at<float>(i,j);
                    
                }
            }
        }
        //convert hist to mat
        for( int h = 0; h < NBINS_EDGESDIR; h++ )
        {
            descriptor.at<float>(0,h) = descriptor.at<float>(0,h) / maxProb;
            //float val =  descriptor.at<float>(0,h);
            //printf("EDGESDIR: %d %f MAX:%f \n",h,val,maxProb);
        }

        return descriptor;
    }

    Mat descriptorsSEMANTIC(int SEMANTIC_LABELS = 60)
    {
        
        Mat descriptor = Mat::zeros(1, (SEMANTIC_LABELS), CV_32FC1);
        //MatND histN;
        //normalize(_labelSegmentation, histN);
        
        labelSet labels(SEMANTIC_LABELS);
        printf("---------------------\n");
        printf("label segmentation\n");//*/
        //convert _labelSegmentation
        for( int h = 0; h < (SEMANTIC_LABELS); h++ )
        {
            float binVal = _labelSegmentation.at<float>(h);
            if (binVal != 0) printf("\t%d %s %f \n",
                                    h,
                                    labels.getLabel(h).c_str(),
                                    binVal);//*/
            descriptor.at<float>(0,h)=binVal;
        }
        
      /* // printf("neighour segmentation\n");
        //convert _labelFirstSegmentation
        for( int h = (SEMANTIC_LABELS); h < 2*(SEMANTIC_LABELS); h++ )
        {
            float binVal = _labelFirstSegmentation.at<float>(h-(SEMANTIC_LABELS));
           /* if (binVal != 0) printf("\t%d %s %f \n",
                   h-(SEMANTIC_LABELS),
                   labels.getLabel(h-(SEMANTIC_LABELS)).c_str(),
                   binVal);*/
        /*    descriptor.at<float>(0,h)=binVal;
        }
       // printf("---------------------\n");*/
        
        //histN.release();
        return descriptor;
    }
    
    Mat descriptorsCONTEXT(int SEMANTIC_LABELS = 60)
    {
        
        Mat descriptor = Mat::zeros(1, (SEMANTIC_LABELS), CV_32FC1);
        
        labelSet labels(SEMANTIC_LABELS);
        
         printf("neighour segmentation\n");

        for( int h = 0; h < (SEMANTIC_LABELS); h++ )
        {
            float binVal = _labelFirstSegmentation.at<float>(h);
             if (binVal != 0) printf("\t%d %s %f \n",
             h,
             labels.getLabel(h).c_str(),
             binVal);//*/
            descriptor.at<float>(0,h)=binVal;
        }
        
        return descriptor;
    }
    
    Mat descriptorsLINES(Mat image, int BINS = 8)
    {
        Mat descriptor = Mat::zeros(1, BINS, CV_32FC1);
        
        Mat img;
        cvtColor(image, img, CV_BGR2GRAY);
        
        Mat src, src_gray;
        Mat dst, detected_edges;
        
        //int edgeThresh = 1;
        int lowThreshold = 10;
        // int const max_lowThreshold = 100;
        int ratio = 3;
        int kernel_size = 3;
        /// Reduce noise with a kernel 3x3
        blur( image, detected_edges, Size(3,3) );
        
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
        
        
        vector<Vec4i> lines;
        HoughLinesP( detected_edges, lines, 1, CV_PI/180, 80, 30, 10);// , minLineLength=0);
        cvtColor(detected_edges,detected_edges,CV_GRAY2BGR);
        for( size_t i = 0; i < lines.size(); i++ )
        {
            //filter by size
            
            float dis = sqrt(fabs((float)(lines[i][0]-lines[i][2]) * (lines[i][0]-lines[i][2])) +
                             fabs((float)(lines[i][1]-lines[i][3]) * (lines[i][1]-lines[i][3])));
            printf("%f\n",dis);
            
            if (dis < 35)
                
                line( detected_edges, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255,0,0), 3, 8 );
            
            //imshow("lines",detected_edges); waitKey(0);
        }

        /// Using Canny's output as a mask, we display our result
        
        return  detected_edges;
        
    }//descriptorsLINES
    
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
