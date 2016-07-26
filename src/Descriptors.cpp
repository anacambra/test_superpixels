//
//  descriptors.cpp
//  tests
//
//  Created by Ana B. Cambra on 06/05/16.
//  Copyright (c) 2016 Ana B. Cambra. All rights reserved.
//

#include <stdio.h>
//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "superpixel.cpp"

#define _DEBUG_DES 0

class Descriptors
{
    
    
    public:
    
    
    
    /////////////////////////////////////////////////////////////////////////////////////
    //files
    static void descriptors2file(FILE *fout, float fid, Mat desID, float acc)
    {
        fwrite(&fid,sizeof(float),1,fout);
        
        for(int d=0; d < desID.cols; d++)
        {
            float val = desID.at<float>(d);
            fwrite(&val,sizeof(float),1,fout);
            //if (fid == 0.0) printf("id: %f descriptor: %d %f %f\n",fid,d,(float)desID.at<float>(d),val);
        }
        
        fwrite(&acc,sizeof(float),1,fout);
    }
    
    static bool file2descriptors(string file, int size, Mat *descriptors, Mat *accText)
    {
        FILE *fid = fopen(file.c_str(),"rb");
        if (fid == NULL)
        return false;
        
        int sizeID = size + 2;
        float data[sizeID];
        int id=0;
        
        while(!feof(fid))
        {
            size_t d = fread(data, sizeof(data[0]), sizeID, fid);
            //data: id des acc
            if ((int)d > 0)
            {
                for (int i = 1; i <= size; i++)
                {
                    descriptors->at<float>(id,i-1)= (float) data[i];
                    //printf("id: %d leidos: %d %f %f\n",id,(int)d,data[i],descriptors->at<float>(id,i-1));
                }
                accText->at<float>(id)= data[size+1];
                id = id + 1;
            }
        }
        
        fclose(fid);
        return true;
    }

    
    /////////////////////////////////////////////////////////////////////////////////////

    
    
    ///////////////////////////////////////////////////////////////////////////////////
    
    
    /********************************/
    /*  LAB                         */
    /********************************/
    static Mat descriptorsLAB(Mat image,Mat maskLAB,int NBINS_L = 7, int NBINS_AB = 7)
    {
        MatND hist_l;
        MatND hist_a;
        MatND hist_b;
        
        Mat descriptor = Mat::zeros(1, NBINS_L+NBINS_AB+NBINS_AB, CV_32FC1);
        
        Mat image_out;
        cvtColor(image, image_out, CV_BGR2Lab);
        
        vector<Mat> spl;
        split(image_out,spl);
        
        int _numPixels;
        
        if ( maskLAB.data != NULL )
            _numPixels = countNonZero(maskLAB);
        else
        {
            _numPixels = image.cols*image.rows;
        }
        
        //L <- L * 255/100 ; a <- a + 128 ; b <- b + 128
        
        //0 < L < 100
        int nbins = NBINS_L; // levels
        int hsize[] = { nbins }; // just one dimension
        
        float range[] = { 0, (const float)(100)};
        const float *ranges[] = { range };
        int chnls[] = {0};
        
        spl[0]=spl[0] * (NBINS_L - 1)/255;
        
        calcHist(&spl[0], 1, chnls, maskLAB, hist_l,1,hsize,ranges);
        
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
        
        calcHist(&spl[1], 1, chnls, maskLAB, hist_a,1,hsizeA,rangesA);
        
        for(int b = 0; b < nbinsA; b++ )
        {
            float binVal = hist_a.at<float>(b);
            descriptor.at<float>(0,NBINS_L+b)= binVal /(float)_numPixels;
        }
        
        //-128 < b < 127
        calcHist(&spl[2], 1, chnls, maskLAB, hist_b,1,hsizeA,rangesA);
        
        for(int b = 0; b < nbinsA; b++ )
        {
            float binVal = hist_b.at<float>(b);
            descriptor.at<float>(0,NBINS_L + NBINS_AB + b)= binVal /(float)_numPixels;
        }
        
        if (_DEBUG_DES == 1 && maskLAB.data == NULL)
        {
            imshow("No mask image",image);
            imshow("No mask Hist L",SuperPixel::paintHistogram(hist_l));
            imshow("No mask Hist A",SuperPixel::paintHistogram(hist_a));
            imshow("No mask Hist B",SuperPixel::paintHistogram(hist_b));
        }else if (_DEBUG_DES == 1)
        {
            imshow("image",image);
            imshow("mask",maskLAB);
            imshow("Hist L",SuperPixel::paintHistogram(hist_l));
            imshow("Hist A",SuperPixel::paintHistogram(hist_a));
            imshow("Hist B",SuperPixel::paintHistogram(hist_b));
            
        }
        
        spl[0].release();
        spl[1].release();
        spl[2].release();
        hist_l.release();
        hist_a.release();
        hist_b.release();
        image_out.release();
        
        return descriptor.clone();
    }//descriptorsLAB
    
    /********************************/
    /*  EDGES                       */
    /********************************/
    static Mat descriptorsEDGES(Mat image, Mat maskEDGES, int BINS = 4, int mode = 1 )
    {
        if (image.data == NULL)
        {
            printf("ERROR: image edges not found\n");
            return Mat();
        }
        
        //mode = 0: histogram
        //mode = 1: moments mean stdDes skew kurtosis
        //mode = 2: hist + moments
        
        int momentsSIZE = 4;
        //image is BGR float
        Mat out;
        
        int _numPixels;
        
        if ( maskEDGES.data != NULL )
            _numPixels = countNonZero(maskEDGES);
        else
        {
            _numPixels = image.cols*image.rows;
        }
        
        
        Mat descriptor = Mat::zeros(1, BINS, CV_32FC1);
        cvtColor(image, out, CV_BGR2GRAY);
        out.convertTo(out, CV_32FC1,1.0/255.0,0);
        
        
        if ( maskEDGES.data != NULL && (out.rows != maskEDGES.rows || out.cols != maskEDGES.cols))
        {
            resize(out, out, Size(maskEDGES.cols,maskEDGES.rows));
            cout<< "Warning! RESIZE image in DescriptorEDGES";
            // resize(maskEDGES, maskEDGES, Size(out.rows,out.cols));
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
            
            calcHist(&out, 1, chnls, maskEDGES, hist,1,hsize,ranges);
            
            for(int b = 0; b < nbins; b++ )
            {
                float binVal = hist.at<float>(b);
                if (mode == 0 || mode == 2) descriptor.at<float>(0,b)=(binVal / (float)_numPixels);
                // printf("%d \t %f\n",b,binVal);
            }
            
            if (_DEBUG_DES == 1 && maskEDGES.data == NULL)
                imshow("Hist EDGES no mask",SuperPixel::paintHistogram(hist));
            else if (_DEBUG_DES == 1) imshow("Hist EDGES",SuperPixel::paintHistogram(hist));
            
            hist.release();
        }
        
        //MOMENTS
        if (mode == 1 || mode == 2)
        {

            Mat mask = Mat::zeros(out.rows,out.cols,CV_32FC1);
            if (maskEDGES.data == NULL) {
                mask = out.clone();
            }else//*/
                out.copyTo(mask, maskEDGES);
            
            double min, max;
            minMaxLoc(mask, &min, &max);
            Scalar     mean, stddev;
            meanStdDev ( mask, mean, stddev, maskEDGES );
            
            float mc3,mc4;
            
            mc3=0;
            mc4=0;
            
            for (int x = 0; x < mask.rows; x++)
            {
                for (int y = 0; y < mask.cols; y++)
                {
                    if (mask.at<uchar>(x,y) != 0)
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
            
            if (_DEBUG_DES == 1)
            {
               // imshow("descriptor edges",mask);
                printf("Superpixel::EDGES [%f,%f] mean %f stdDev %f s %f k %f \n", mc3,mc4, mean.val[0],stddev.val[0],s,k);
            }
            
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
    /********************************/
    /*  EDDIR                       */
    /********************************/
    static  Mat descriptorsEDGESDIR(Mat edges,Mat edgesDIR, Mat mask,int NBINS_EDGESDIR = 8)
    {
        Mat descriptor = Mat::zeros(1, NBINS_EDGESDIR, CV_32FC1);
        float maxProb = 0.0;
        
        for( int h = 0; h < NBINS_EDGESDIR; h++ )
        descriptor.at<float>(0,h)=0.0;
        
        //edges [0-1] probabiltities Max= 1.0*_numPixels
        cvtColor(edges, edges, CV_BGR2GRAY);
        edges.convertTo(edges, CV_32FC1,1.0/255.0,0);
        
        if (edges.rows != mask.rows || edges.cols != mask.cols)
        {
            resize(edges, edges, Size(mask.cols,mask.rows));
        }
        if (edgesDIR.rows != mask.rows || edgesDIR.cols != mask.cols)
        {
            resize(edgesDIR, edgesDIR, Size(mask.cols,mask.rows));
        }
       
        //edgesDir [0-3.14]
        int numPixels=0;
        
        for (int i=0; i< edgesDIR.rows; i++)
        {
            for (int j=0; j< edgesDIR.cols; j++)
            {
                if (mask.at<uchar>(i,j) != 0)
                {
                    int b = (int)(edgesDIR.at<float>(i,j)/(3.141593/(float)NBINS_EDGESDIR));
                    descriptor.at<float>(0,b) += edges.at<float>(i,j);
                    maxProb += edges.at<float>(i,j);
                    numPixels++;
                }
            }
        }
        
        //convert hist to mat
        for( int h = 0; h < NBINS_EDGESDIR; h++ )
        {
            if (maxProb != 0.0)
            descriptor.at<float>(0,h) = descriptor.at<float>(0,h) / maxProb;
        }
        
        return descriptor;
    }//descriptorsEDDIR
    
    /********************************/
    /*  HOG                       */
    /********************************/
    static Mat DescriptorHOG(Mat img, Point centroide, Size window = Size(80,80))
    {
        HOGDescriptor hog;
        hog.winSize = window;
        //only supported values
        hog.blockSize = Size(16,16);
        hog.blockStride = Size(8,8);
        hog.cellSize = Size(8,8);
        hog.nbins = 9;
        
        int BINS_HOG=2916; // = (5+4)*(5+4) bloques * 4 celdas por bloque * 9 bins por celda
        
        Mat des = Mat::zeros(1, BINS_HOG, CV_32FC1);
        
        Mat gray;
        vector< Point > location;//location up corner centroide
        vector< float > descriptors;
        
        Point myPoint;
        location.push_back(myPoint);
        location[0].x = centroide.x - window.width/2;
        if (location[0].x< 0) location[0].x=0;
        if (location[0].x + window.width > img.cols) location[0].x=img.cols - window.width;
        
        location[0].y = centroide.y - window.height/2;;//*/
        if (location[0].y< 0) location[0].y=0;
        if (location[0].y + window.height > img.rows) location[0].y=img.rows - window.height;
        
        
        cvtColor( img, gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors,window, Size( 0, 0 ), location );
        
        
        for (std::vector<float>::const_iterator i = descriptors.begin(); i != descriptors.end(); ++i)
        {
            des.at<float>(0,(int)(i - descriptors.begin())) = *i;
            //std::cout << (i - descriptors.begin()) << ": "<< *i ;
            // std::cout << (i - descriptors.begin()) << " "<<  des.at<float>(0,(int)(i - descriptors.begin())) << std::endl;
            
        }
        
        /* for (std::vector<Point>::const_iterator j = location.begin(); j != location.end(); ++j)
         {
         circle(img,centroide,3,CV_RGB(255,0,0),-1,8,0);
         rectangle(img, Rect((*j).x,(*j).y,window.width,window.height),CV_RGB(0,0,255),2,8,0);
         std::cout <<  (*j).x << std::endl;
         
         }//*/
        
        gray.release();
        return des;
    }
    
    static Mat DescriptorHOG(Mat img, Size window = Size(80,80))
    {
        //resize image
        Mat gray;
        resize(img,gray,window);
        
        
        HOGDescriptor hog;
        hog.winSize = window;
        //only supported values
        hog.blockSize = Size(16,16);
        hog.blockStride = Size(8,8);
        hog.cellSize = Size(8,8);
        hog.nbins = 9;
        
        int BINS_HOG=2916; // = (5+4)*(5+4) bloques * 4 celdas por bloque * 9 bins por celda
        
        Mat des = Mat::zeros(1, BINS_HOG, CV_32FC1);
        
        
        vector< Point > location;//location up corner centroide
        vector< float > descriptors;
        
        
        cvtColor( gray, gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors,window, Size( 0, 0 ), location );
        
        
        for (std::vector<float>::const_iterator i = descriptors.begin(); i != descriptors.end(); ++i)
        {
            des.at<float>(0,(int)(i - descriptors.begin())) = *i;
            //std::cout << (i - descriptors.begin()) << ": "<< *i ;
            //std::cout << (i - descriptors.begin()) << " "<<  des.at<float>(0,(int)(i - descriptors.begin())) << std::endl;
            
        }
        
        /* for (std::vector<Point>::const_iterator j = location.begin(); j != location.end(); ++j)
         {
         circle(img,centroide,3,CV_RGB(255,0,0),-1,8,0);
         rectangle(img, Rect((*j).x,(*j).y,window.width,window.height),CV_RGB(0,0,255),2,8,0);
         std::cout <<  (*j).x << std::endl;
         
         }//*/
        
        gray.release();
        return des;
    }
  
    
    /* RGB */
    static Mat descriptorsRGB(Mat image, Mat _mask, int BINS = 256)
    {
        // return Mat(1,256*3,CV_32FC1)
        Mat descriptor = Mat::zeros(1, BINS*3, CV_32FC1);
        int _numPixels = countNonZero(_mask);
        
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
    
    
    /* PEAKS */
    static Mat descriptorsPEAKS(Mat image, Mat _mask, int BINS = 256)
    {
        Mat descriptor = Mat::zeros(1, BINS, CV_32FC1);
        int _numPixels = countNonZero(_mask);
        
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
        
        if (_DEBUG_DES == 1) imshow("Hist GRAY",SuperPixel::paintHistogram(hist));
        hist.release();
        img.release();
        return descriptor;
    }//descriptorsPEAKS
    
    
    /*HoG*/
   

    
    

};