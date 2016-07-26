//
//  hog.cpp
//  tests
//
//  Created by Ana B. Cambra on 07/06/16.
//  Copyright (c) 2016 Ana B. Cambra. All rights reserved.
//

#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

using namespace cv;

class hog
{
    //void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size )
    static void compute_hog(Mat  img, const Size & size )
    {
        HOGDescriptor hog;
        hog.winSize = size;
        //only supported values
        hog.blockSize = Size(16,16);
        hog.blockStride = Size(8,8);
        hog.cellSize = Size(8,8);
        hog.nbins = 9;
        Mat gray;
        vector< Point > location;
        vector< float > descriptors;
        Point myPoint;
        location.push_back(myPoint);
        location[0].x = 0;
        location[0].y = 0;//*/
        
        // vector< Mat >::const_iterator img = img_lst.begin();
        // vector< Mat >::const_iterator end = img_lst.end();
        // for( ; img != end ; ++img )
        // {
        cvtColor( img, gray, COLOR_BGR2GRAY );
        
        
        hog.compute( gray, descriptors,size, Size( 0, 0 ), location );
        
        
        for (std::vector<float>::const_iterator i = descriptors.begin(); i != descriptors.end(); ++i)
            std::cout << (i - descriptors.begin()) << ": "<< *i << std::endl;
        
        
        for (std::vector<Point>::const_iterator j = location.begin(); j != location.end(); ++j)
        {
            circle(img,*j,3,CV_RGB(255,0,0),-1,8,0);
            std::cout <<  (*j).x << std::endl;
            /*imshow( "img", img);
            waitKey( 0 );*/
        }
        
        
        // gradient_lst.push_back( Mat( descriptors ).clone() );
        //#ifdef _DEBUG
        //imshow( "gradient", get_hogdescriptor_visu( img.clone(), descriptors, size ) );
         imshow( "img", img);
        //waitKey( 0 );
        //#endif
        //}
    }
    
    
    
    static Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
    {
        const int DIMX = size.width;
        const int DIMY = size.height;
        float zoomFac = 10;
        Mat visu;
        resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );
        
        int cellSize        = 8;
        int gradientBinSize = 9;
        float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?
        
        // prepare data structure: 9 orientation / gradient strenghts for each cell
        int cells_in_x_dir = DIMX / cellSize;
        int cells_in_y_dir = DIMY / cellSize;
        float*** gradientStrengths = new float**[cells_in_y_dir];
        int** cellUpdateCounter   = new int*[cells_in_y_dir];
        for (int y=0; y<cells_in_y_dir; y++)
        {
            gradientStrengths[y] = new float*[cells_in_x_dir];
            cellUpdateCounter[y] = new int[cells_in_x_dir];
            for (int x=0; x<cells_in_x_dir; x++)
            {
                gradientStrengths[y][x] = new float[gradientBinSize];
                cellUpdateCounter[y][x] = 0;
                
                for (int bin=0; bin<gradientBinSize; bin++)
                    gradientStrengths[y][x][bin] = 0.0;
            }
        }
        
        // nr of blocks = nr of cells - 1
        // since there is a new block on each cell (overlapping blocks!) but the last one
        int blocks_in_x_dir = cells_in_x_dir - 1;
        int blocks_in_y_dir = cells_in_y_dir - 1;
        
        // compute gradient strengths per cell
        int descriptorDataIdx = 0;
        int cellx = 0;
        int celly = 0;
        
        for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
        {
            for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
            {
                // 4 cells per block ...
                for (int cellNr=0; cellNr<4; cellNr++)
                {
                    // compute corresponding cell nr
                    cellx = blockx;
                    celly = blocky;
                    if (cellNr==1) celly++;
                    if (cellNr==2) cellx++;
                    if (cellNr==3)
                    {
                        cellx++;
                        celly++;
                    }
                    
                    for (int bin=0; bin<gradientBinSize; bin++)
                    {
                        float gradientStrength = descriptorValues[ descriptorDataIdx ];
                        descriptorDataIdx++;
                        
                        gradientStrengths[celly][cellx][bin] += gradientStrength;
                        
                    } // for (all bins)
                    
                    
                    // note: overlapping blocks lead to multiple updates of this sum!
                    // we therefore keep track how often a cell was updated,
                    // to compute average gradient strengths
                    cellUpdateCounter[celly][cellx]++;
                    
                } // for (all cells)
                
                
            } // for (all block x pos)
        } // for (all block y pos)
        
        
        // compute average gradient strengths
        for (celly=0; celly<cells_in_y_dir; celly++)
        {
            for (cellx=0; cellx<cells_in_x_dir; cellx++)
            {
                
                float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
                
                // compute average gradient strenghts for each gradient bin direction
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
                }
            }
        }
        
        // draw cells
        for (celly=0; celly<cells_in_y_dir; celly++)
        {
            for (cellx=0; cellx<cells_in_x_dir; cellx++)
            {
                int drawX = cellx * cellSize;
                int drawY = celly * cellSize;
                
                int mx = drawX + cellSize/2;
                int my = drawY + cellSize/2;
                
                rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), Scalar(100,100,100), 1);
                
                // draw in each cell all 9 gradient strengths
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float currentGradStrength = gradientStrengths[celly][cellx][bin];
                    
                    // no line to draw?
                    if (currentGradStrength==0)
                        continue;
                    
                    float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
                    
                    float dirVecX = cos( currRad );
                    float dirVecY = sin( currRad );
                    float maxVecLen = (float)(cellSize/2.f);
                    float scale = 2.5; // just a visualization scale, to see the lines better
                    
                    // compute line coordinates
                    float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                    float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                    float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                    float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                    
                    
                    // draw gradient visualization
                    line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), Scalar(0,255,0), 1);
                    
                } // for (all bins)
                
            } // for (cellx)
        } // for (celly)
        
        
        // don't forget to free memory allocated by helper data structures!
        for (int y=0; y<cells_in_y_dir; y++)
        {
            for (int x=0; x<cells_in_x_dir; x++)
            {
                delete[] gradientStrengths[y][x];
            }
            delete[] gradientStrengths[y];
            delete[] cellUpdateCounter[y];
        }
        delete[] gradientStrengths;
        delete[] cellUpdateCounter;
        
        return visu;
        
    } // get_hogdescriptor_visu

    
public:
    static void testHog(Mat img)
    {
        //HOG descriptors
        //Mat  img = imread("/Users/acambra/Desktop/veryDeepNet/test/images/img_1/img_1.jpg_0.png",CV_LOAD_IMAGE_COLOR);
        vector< float >  gradient_lst;
        
        int c,r;
        c= 80;//ceil(double(img.cols/16.0))*16;
        r= 80;//ceil(double(img.rows/16.0))*16;
        resize(img,img,Size(c,r));
        
        compute_hog( img, Size( 80,80) );
        
    }
    
   /* static Mat DescriptorHOG(Mat img, Point centroide, Size window = Size(80,80))
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
        
        location[0].y = centroide.y - window.height/2;
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
        
        for (std::vector<Point>::const_iterator j = location.begin(); j != location.end(); ++j)
        {
            circle(img,centroide,3,CV_RGB(255,0,0),-1,8,0);
            rectangle(img, Rect((*j).x,(*j).y,window.width,window.height),CV_RGB(0,0,255),2,8,0);
            std::cout <<  (*j).x << std::endl;
           
        }
        
        gray.release();
        return des;
    }//*/
};
