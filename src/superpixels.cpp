#ifndef SUPERPIXELS_H
#define SUPERPIXELS_H

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

#include "superpixel.cpp"
#include "Descriptors.cpp"

#include <time.h>

//vlfeat
#if VLFEAT
#define VLFEAT
#include "vl/generic.h"
#include "vl/slic.h"
#endif

#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "lib_slic/SLIC.h"

#define PI 3.1416
#define SP_MASK 1
#define BB_SP_MASK 0
#define BB_NEIG_MASK 2


using namespace std;

class SuperPixels
{
 
    Mat _image; // original image in color BGR
    Mat _ids; // superpixels ids  CV_8UC1
    Mat _sobel; //MASK superpixel boundaries  UCHAR
    Mat _labelsInput; //input labeling
    
    Mat _labels; // labeling CV_32FC1
    
    // superpixels params
    int _TAM_SP = 35;
    int _NUM_MAX_SP = 700;
    
    int NUMLABELS;
    
    unsigned char _DEBUG = 1;
    
public:
    
    SuperPixel *_arraySP;
    int maxID;
    
    //utilsCaffe *_caffe;
    
    //time
    
    float timeSuperpixels = 0.0;
    float timeLAB = 0.0;
    float timeRGB = 0.0;
    float timeEDGES = 0.0;
    float timeEDGESDIR = 0.0;
    float timeCAFFE = 0.0;
    float timeSEMANTIC = 0.0;
    
    
    SuperPixels()
    {   maxID=0;
    }
    ~SuperPixels()
    {
         delete[] _arraySP;
        _image.release();
        _ids.release();
        _sobel.release();
        _labels.release();
        _labelsInput.release();
       //delete[] _arraySP;
        //delete _caffe;
    }
    
    static Rect boundingbox(Mat image)
    {
        Mat nonZeroCoordinates;
        
        if (countNonZero(image) == 0)
            return Rect(0,0,0,0);
        
        
        findNonZero( image, nonZeroCoordinates);
        double minX=image.cols+1, minY=image.rows+1, maxX=-1,maxY=-1;
        
        for (int i = 0; i < nonZeroCoordinates.total(); i++ )
        {
            if (nonZeroCoordinates.at<Point>(i).x <= minX) minX = nonZeroCoordinates.at<Point>(i).x;
            else if (nonZeroCoordinates.at<Point>(i).x >= maxX) maxX =  nonZeroCoordinates.at<Point>(i).x;
            
            if (nonZeroCoordinates.at<Point>(i).y <= minY) minY = nonZeroCoordinates.at<Point>(i).y;
            else if (nonZeroCoordinates.at<Point>(i).y >= maxY) maxY =  nonZeroCoordinates.at<Point>(i).y;
        }
        
        
        return Rect(minX,minY,(maxX-minX)+1, (maxY-minY)+1);
    }
    
    Mat getMask(int id)
    {
        return _arraySP[id].getMask();
    }
    
    Mat getMaskBB(int id)
    {
        Mat mask =_arraySP[id].getMask().clone();
        rectangle(mask,boundingbox(mask), 255,-1,CV_AA,0);
        return mask;
    }
    
    Mat getMaskNeigboursBB(int id)
    {
        set<int> neig = _arraySP[id].getFirstNeighbours();
        
        std::set<int>::iterator it;
        
        Mat maskN = Mat::zeros(_image.rows,_image.cols, CV_8UC1);
        for (it=neig.begin(); it!=neig.end(); ++it)
        {
           //concat mask
            bitwise_or(maskN, _arraySP[*it].getMask(),maskN);
        }
        
        //paint white BB neigbours
        rectangle(maskN,boundingbox(maskN), 255,-1,CV_AA,0);
        
        //paint black superpixel
        Mat maskID= getMaskBB(id);
        
        Mat mask;
        bitwise_xor(maskN,maskID,maskN);

        return maskN.clone();
    }
    
    void activeDEBUG(){ _DEBUG = 1;}
    void desactiveDEBUG(){ _DEBUG = 0;}
    
    int numPixels(int id)
    {
        return _arraySP[id].getNumPixels();
    }
    
    Mat getMaskNeigbours(int id)
    {
        set<int> neig = _arraySP[id].getFirstNeighbours();
        int ln=0;
        
        std::set<int>::iterator it;
        
        Mat maskN = Mat::zeros(_image.rows,_image.cols, CV_8UC1);
        for (it=neig.begin(); it!=neig.end(); ++it)
        {
            ln = _arraySP[id].addHistogramLabelSegmentation(_arraySP[*it].getLabelSegmentation());
            //printf("id: %d neig: %d l=%d  %d\n",id1,*it,ln,(int)neig.size());//getchar();
            //concat mask
            bitwise_or(maskN, _arraySP[*it].getMask(),maskN);
        }
    
        return maskN;
        
    }
    
    /*************************************************************************************
     * SuperPixels: obtain superpixels of an image (path)
     *  load _image (path) and obtain its superpixels
     *      if image_TAM_SP.sp exits -> loadFile
     *      else SLIC & save TAM_SP.sp
     *
     */
    SuperPixels(string path)
    {
        clock_t start = clock();
        maxID=0;
        
        //read image
        try{
            _image = imread(path,CV_LOAD_IMAGE_COLOR);
            
            if(_image.data == NULL)
            {
                printf("Image %s not found\n",path.c_str());
                return;
            }
            else
               if (_DEBUG == 1) printf("Mat _image CV_8UC1 rows %d cols %d\n",_image.rows,_image.cols);
            
            _ids= Mat::zeros(_image.rows,_image.cols,CV_32FC1);
            _sobel= Mat::zeros(_image.rows,_image.cols,CV_8UC1);
            _labels= Mat::ones(_ids.rows,_ids.cols,CV_32FC1)*-1;
        }
        catch(int e)
        {
            printf("Image %s not found\n",path.c_str());
            return;
        }
        
        size_t found = path.find_last_of(".");
        std::string name = path.substr(0,found) + "_" + std::to_string(_TAM_SP)+".sp";
        FILE *f = fopen(name.c_str(),"r");
        
        if (f!=NULL)
        {
            fclose(f);
            
            start = clock();
            
            loadSuperPixels(name);
            
            timeSuperpixels = (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
            
            if (_DEBUG == 1) printf("**** TIME: load Superpixels: %f seconds\n",(float) (((double)(clock() - start)) / CLOCKS_PER_SEC) );
        }
        
        else
        {
            fclose(f);
            
            if (_DEBUG == 1) start = clock();
            
            //To-Do: ./slic_cli  --input test_image/ --contour --superpixels 100 --csv
            calculateSLICSuperpixels(_image);
            
            if (_DEBUG == 1) printf("**** TIME: calculate Superpixels: %f seconds\n ",(float) (((double)(clock() - start)) / CLOCKS_PER_SEC) );
            
            //save superpixels in a file
            superpixels2file(name);
            
        }

        _arraySP = new SuperPixel[maxID+1];
        return;
    }//SuperPixels

    
    /*****************************************************************/
    /*****************************************************************/
    void setNUMLABELS(int n)
    {
        NUMLABELS = n;
    }
    
    Mat getImageLabels() {return _labels;}
    Mat getImage()
    {
        if(_image.data == NULL)
        {
            return Mat::zeros(100, 100, CV_8U);
        }
        else
            return _image;
    }
    
    Mat getImageSuperpixels(){
        
        Mat im= _image.clone();
        Scalar* color = new cv::Scalar( 0, 0, 255 );
        im.setTo(*color,_sobel);
        delete color;
        return im;
    }
    
    Mat paintSuperpixel(Mat image, int id, Scalar *color = new cv::Scalar(0,0,255)){
        
        Mat im =image.clone();
       // Scalar* color = new cv::Scalar( 0, 0, 255 );
        im.setTo(*color,_arraySP[id].getMask());
        /*imshow("mask",this->getMask(id));
        waitKey(0);
        destroyAllWindows();*/
        delete color;
        return im;
    }//paintSuperpixel
    
    Mat paintNeighboursSuperpixel(Mat image, int id){
        
        Mat im =image.clone();
        Scalar* color = new cv::Scalar( 0, 0, 255 );
        im.setTo(*color,_arraySP[id].getMask());
        
        // its neighbours
        color = new cv::Scalar( 0, 255, 0 );
        set<int> neig = _arraySP[id].getFirstNeighbours();
        
        std::set<int>::iterator it;
        for (it=neig.begin(); it!=neig.end(); ++it)
        {
            im.setTo(*color,_arraySP[*it].getMask());
        }
        
        color = new cv::Scalar( 255, 0, 0 );
        neig = _arraySP[id].getSecondNeighbours();
        
        for (it=neig.begin(); it!=neig.end(); ++it)
        {
            im.setTo(*color,_arraySP[*it].getMask());
        }
        delete color;
    
        return im;
    }//paintNeighboursSuperpixel
    
    /*************************************************************************************
     * calculateSLICSuperpixels
     *
     *      mat: BGR 8 bit channels
     *      fill : _ids, maxID
     *      limit total number of superpixels (maxID+1)
     */
    void calculateSLICSuperpixels(Mat mat)
    {
        // Convert matrix to unsigned int array.
        unsigned int* image = new unsigned int[mat.rows*mat.cols];
        unsigned int value = 0x0000;
        
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                
                int b = mat.at<cv::Vec3b>(i,j)[0];
                int g = mat.at<cv::Vec3b>(i,j)[1];
                int r = mat.at<cv::Vec3b>(i,j)[2];
                
                value = 0x0000;
                value |= (0x00FF0000 & (r << 16));
                value |= (0x0000FF00 & (b << 8));
                value |= (0x000000FF & g);
                
                image[j + mat.cols*i] = value;
            }
        }
        
        SLIC slic;
        
        int* segmentation = new int[mat.rows*mat.cols];
        int numberOfLabels = 0;
        
        //timer.restart();
        int superpixels = sqrt(float(mat.rows*mat.cols)/_TAM_SP);
        if (superpixels > _NUM_MAX_SP) superpixels = _NUM_MAX_SP;
    
        double compactness = 40;
        bool perturbseeds = false;
        int iterations = 10;
        
        slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(image, mat.cols, mat.rows, segmentation, numberOfLabels, superpixels, compactness, perturbseeds, iterations);
        
        // Convert labels.
        int** labels = new int*[mat.rows];
        for (int i = 0; i < mat.rows; ++i) {
            labels[i] = new int[mat.cols];
            
            for (int j = 0; j < mat.cols; ++j) {
                labels[i][j] = segmentation[j + i*mat.cols];
                if (labels[i][j] >= maxID) maxID=labels[i][j];
                //printf("label %d \n",labels[i][j]);
                _ids.at<float>(i,j)=(float)labels[i][j];
            }
        }
        
        delete[] image;
        delete(segmentation);
        delete(labels);
        
    }

#if VLFEAT
    void calculateSLICSuperpixelsVLFEAT(Mat mat){
        // The matrix 'mat' will have 3 8 bit channels
        // corresponding to BGR color space.
        
        // Convert image to one-dimensional array.
        float* image = new float[mat.rows*mat.cols*mat.channels()];
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                // Assuming three channels ...
                image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
                image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
                image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
            }
        }
        
        // The algorithm will store the final segmentation in a one-dimensional array.
        vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
        vl_size height = mat.rows;
        vl_size width = mat.cols;
        vl_size channels = mat.channels();
        
        // The region size defines the number of superpixels obtained.
        // Regularization describes a trade-off between the color term and the
        // spatial term.
        
        //LIMIT NUM SUPERPIXELS with SIZE
        int numSup = mat.rows*mat.cols /(_TAM_SP*_TAM_SP);
        vl_size region;
        
        if (numSup > _NUM_MAX_SP)
        {
            //change superpixel size
            region = sqrt(float(mat.rows*mat.cols)/_NUM_MAX_SP);
        }
        else
            region = _TAM_SP;
        
        if (_DEBUG == 1) printf("* Default: TAM: %d MAX: %d \n* %d pixels: TAM: %d MAX: %d\n", _TAM_SP,_NUM_MAX_SP, mat.rows*mat.cols,(int)region,numSup);

        float regularization = 10000;
        vl_size minRegion = (int)region / 2;
        
        vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);
        
        // Convert segmentation.
        int** labels = new int*[mat.rows];
        for (int i = 0; i < mat.rows; ++i) {
            labels[i] = new int[mat.cols];
            
            for (int j = 0; j < mat.cols; ++j) {
                labels[i][j] = (int) segmentation[j + mat.cols*i];
            }
        }//for
        
        // Compute a contour image: this actually colors every border pixel
        // red such that we get relatively thick contours.
        int label = 0;
        int labelTop = -1;
        int labelBottom = -1;
        int labelLeft = -1;
        int labelRight = -1;
        
        //Fill _ids
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                
                label = labels[i][j];

                if (label >= maxID) maxID=label;
                //printf("label %d \n",label);
                _ids.at<float>(i,j)=(float)label;
                
                
                labelTop = label;
                if (i > 0) {
                    labelTop = labels[i - 1][j];
                }
                
                labelBottom = label;
                if (i < mat.rows - 1) {
                    labelBottom = labels[i + 1][j];
                }
                
                labelLeft = label;
                if (j > 0) {
                    labelLeft = labels[i][j - 1];
                }
                
                labelRight = label;
                if (j < mat.cols - 1) {
                    labelRight = labels[i][j + 1];
                }
                
                if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
                    mat.at<cv::Vec3b>(i, j)[0] = 0;
                    mat.at<cv::Vec3b>(i, j)[1] = 0;
                    mat.at<cv::Vec3b>(i, j)[2] = 255;
                }
            }
        }//for
        
    }//calculateSLICSuperpixels
#endif    
    
    /*************************************************************************************
     * superpixels2file
     */
    void superpixels2file(string nameFile)
    {
        FILE *f;
        int w=0,h=0;
        try{
            f = fopen(nameFile.c_str(),"wb");
            h=_image.rows; w=_image.cols;
            
            for(int i=0;i<h;i++)
                for (int j=0; j<w; j++)
                {
                    int id =(int) _ids.at<float>(i,j);
                    fwrite(&id, sizeof id, 1, f);
                }
            
            fclose(f);
            
        }catch(int e)
        {
            printf("Exception!");
        }
        
    }//superpixels2file
    
    
    /*************************************************************************************
     * loadSLICSuperpixels
     *
     */
    void loadSuperPixels(string path)
    { 
        FILE *f;
        int w=0,h=0;
        maxID = 0;
        
        //ifstream file (path);
        //string current_line;
        
       /* int i=0;
        int j=0;
        
        while(getline(file, current_line)){
            // Now inside each line we need to seperate the cols
            stringstream temp(current_line);
            string single_value;
            while(getline(temp,single_value,',')){
                int id =  atoi(single_value.c_str());
                if (id >= maxID) maxID = id;
                 _ids.at<float>(i,j)=(float)id;
                j=j+1;
                if (j == _image.cols)
                {
                    i= i + 1;
                    j = 0;
                }
            }
        }*/
        
        
        
       try{
            f = fopen(path.c_str(),"rb");
            h=_image.rows; w=_image.cols;
            _ids= Mat::zeros(h,w,CV_32FC1);
            
            for(int i=0;i<h;i++)
                for (int j=0; j<w; j++)
                    if(!feof(f))
                    {
                        int id;
                        fread(&id,sizeof(int),1,f);
                        
                        if (id >= maxID) maxID = id;
                        _ids.at<float>(i,j)=(float)id;
                        
                        //printf("%d %d %s %d\n",i,j,value.c_str() , atoi(value.c_str()));//(int)_ids.at<float>(i,j));
                    }
            
            fclose(f);
            
        }catch(int e){
            printf("Exception!");}
    }
    
    
    /*************************************************************************************
     * calculateBoundariesSuperpixels()
     *
     *
     */
    void calculateBoundariesSuperpixels()
    {
    
        //boundaries
        //SOBEL
        int scale = 1;
        int delta = 0;
        int ddepth =-1;// CV_16S;
        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y,grad;
        
        /// Gradient X
        Sobel( _ids, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        
        /// Gradient Y
        Sobel( _ids, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        
        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        // grad = (grad != 0);
        
        _sobel= (grad != 0);
        
        grad_x.release();
        grad_y.release();
        abs_grad_x.release();
        abs_grad_y.release();
        grad.release();
        
    }//calculateBoundariesSuperpixels
    
    
    /*************************************************************************************
     * initializeSuperpixels()
     * ALL labels = -1
     * obtain set of neighbours
     */
    void initializeSuperpixels()
    {
        //MASKS
        
        if (_ids.data != NULL )
        {
            for(int i=0; i<=maxID; i++ )
            {
                Mat mask_sp;
                mask_sp = (_ids == i);
                _arraySP[i].initialize(i,mask_sp, -1);
                mask_sp.release();
            }
            
             if (_DEBUG == 1) printf("Num superpixels %d\n",maxID);
        }
        else{
            printf("_ids superpixels NULL \n");
            return;
        }
        
        // NEIGHBOUGRS
        
        //first boundaries in _sobel
        for (int x = 0; x < _sobel.rows; x++)
        //for (int x = _sobel.rows; x >=0 ; --x)
        {
            for (int y = 0; y < _sobel.cols; y++)
            //for (int y = _sobel.cols; y >=0 ; --y)
            {
                //printf("x:%d y:%d\n",x,y);
                if ( _sobel.at<uchar>(x,y) == 255) //boundarie
                {
                    int id = (int)_ids.at<float>(x,y);
                    
                    //8-neighbours
                    for (int i=(x-1); i<=(x+1) ; i++)
                    {
                        for (int j=(y-1); j<=(y+1); j++)
                        {

                            if (((x!=i) && (y!=j)) &&
                                (i>=0 && i < _sobel.rows-1) &&
                                (j>=0 && j < _sobel.cols-1)
                                
                                //solo 4 vecinas
                            /* &&(
                             ((i == (x-1)) && (j == (y)))  ||
                             ((i == (x+1)) && (j == (y)))  ||
                             ((i == (x)) && (j == (y-1)))  ||
                             ((i == (x)) && (j == (y+1))) )//*/
                                //solo 4 vecinas sigueintes
                            /* &&(
                             
                             ((i == (x+1)) && (j == (y)))  ||
                             ((i == (x)) && (j == (y+1))) )//*/
                            ){
                                //add neighbours
                                int v = (int)_ids.at<float>(i,j);
                                _arraySP[id].addFirstNeighbour(v);
                                
                                
                            }//if neighbours
                        }//for j
                    }//for i
                }// if boundarie
            }//for y
        }//for x
        
        //second boundaries
        //for (int id=0; id < maxID+1; id++)
        for (int id=maxID; id >= 0; --id)
        {
            set<int> neig1 = _arraySP[id].getFirstNeighbours();
            set<int>::iterator it1;
            for (it1=neig1.begin(); it1!=neig1.end(); ++it1)
            {
                int v1 = (*it1);
                set<int> neig2 = _arraySP[v1].getFirstNeighbours();
                set<int>::iterator it2;
                for (it2=neig2.begin(); it2!=neig2.end(); ++it2)
                {
                    int v2 = (*it2);
                    bool is_in = neig1.find(v2) != neig1.end();
                    if (!is_in && v2 != id)
                        _arraySP[id].addSecondNeighbours(v2);
                }//for it2
            }//for it1
        }//for id

        //MORE DESCRIPTORS
        
    }
    
    //////////
    
    Mat cropSuperpixel(Mat img, int id, float scale = 1, int maskON = 1)
    {
        //1 mask superpixel
        // 0 BB superpixel
        //2 BB N without BB superpixel
        
        if (maskON != BB_NEIG_MASK)
        {
            Mat roi = img(boundingbox(getMask(id))).clone();
            Size s = roi.size();
            if (maskON == SP_MASK)
            {
                Mat mask =_arraySP[id].getMask();
                Mat roiMask = mask(boundingbox(getMask(id))).clone(); //0...255
                cvtColor(roiMask,roiMask,CV_GRAY2BGR);
                
                if (roi.type() != roiMask.type())
                    roi.convertTo(roi, roiMask.type(),255.0,0);
                
                bitwise_and(roi,roiMask, roi);
            }
            
            resize(roi, roi, Size(s.width*scale,s.height*scale));
            
            return roi;
        }else //BB_SP_MASK
        {
            Mat roi = img.clone();
            rectangle(roi,boundingbox(getMask(id)),Scalar(0,0,0),-1,CV_AA,0);
            //imshow("roi",roi);
            
            Mat roiMask = roi(boundingbox(getMaskNeigboursBB(id))).clone();
            Size s = roiMask.size();
            resize(roiMask, roiMask, Size(s.width*scale,s.height*scale));
            
            return roiMask;
        }
    }//cropSuperpixel
    

    /////////////
    Mat calculateDescriptors(int id, Mat image,
                             int mLAB   = 0, int NBINS_L     = 7, int NBINS_AB=7,
                             int mRGB   = 0, int NBINS_RGB   = 256,
                             int mPEAKS = 0, int NBINS_PEAKS = 64,
                             int mEDGES = 0, int NBINS_EDGES = 8, int modeEDGES = 0, Mat edges = Mat::zeros(1,1,CV_32FC3),
                             int mEDDIR = 0, int NBINS_EDDIR = 8, Mat edgesDIR = Mat::zeros(1,1,CV_32FC3),
                             int mCAFFE = 0, string CAFFE_LAYER = "fc7", int NUMCAFFE = 4096,
                             int mSEMANTIC = 0, int SEMANTIC_LABELS = 60,
                             int mCONTEXT = 0, int CONTEXT_LABELS = 60,
                             int mCONTEXT2 = 0, int CONTEXT2_LABELS = 60,
                             int mGLOBAL = 0, int GLOBAL_LABELS = 60,
                             int mCONLAB = 0,
                             int mCONEDGES = 0,
                             int mCONEDDIR = 0,
                             int mHOG =0,   int mCONHOG = 0)
    {
        Mat des;
        
        if (mLAB != 0)
        {
            Mat imageSP = cropSuperpixel(image,id,1,BB_SP_MASK).clone();
            des=Descriptors::descriptorsLAB(imageSP,Mat(),NBINS_L,NBINS_AB).clone();
        }
        
        if (mCONLAB != 0)
        {
            
            if (des.rows != 0)
                hconcat(Descriptors::descriptorsLAB(image,getMaskNeigboursBB(id),NBINS_L,NBINS_AB), des,des);
            else
                des=Descriptors::descriptorsLAB(image,getMaskNeigboursBB(id),NBINS_L,NBINS_AB).clone();
            
        }
        
        if (mRGB != 0)
        {
            if (des.rows != 0)
                hconcat(Descriptors::descriptorsRGB(image,getMask(id),NBINS_RGB), des,des);
            else
                des=Descriptors::descriptorsRGB(image,getMask(id),NBINS_RGB).clone();
        }
        
        if (mPEAKS != 0)
        {
            if (des.rows != 0)
                hconcat(Descriptors::descriptorsPEAKS(image,getMask(id),NBINS_PEAKS), des,des);
            else
                des=Descriptors::descriptorsPEAKS(image,getMask(id),NBINS_PEAKS).clone();
        }
        
        if (mEDGES != 0)
        {
            clock_t start = clock();
            
            if (des.rows != 0)
            {
                Mat imageSP = cropSuperpixel(edges,id,1,BB_SP_MASK).clone();
                hconcat(Descriptors::descriptorsEDGES(imageSP,Mat(),NBINS_EDGES,modeEDGES), des,des);
            }
            else
            {
                Mat imageSP = cropSuperpixel(edges,id,1,BB_SP_MASK).clone();
                des=Descriptors::descriptorsEDGES(imageSP,Mat(),NBINS_EDGES,modeEDGES);
                //_arraySP[id].descriptorsEDGES(edges,NBINS_EDGES,modeEDGES).clone();
            }
            timeEDGES += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
        }
        
        if (mCONEDGES != 0)
        {
            
            if (des.rows != 0)
            {
                hconcat(Descriptors::descriptorsEDGES(edges,getMaskNeigboursBB(id),NBINS_EDGES,modeEDGES), des,des);
            }
            else
            {
                des=Descriptors::descriptorsEDGES(edges,getMaskNeigboursBB(id),NBINS_EDGES,modeEDGES);
            }
        }

        
        if (mEDDIR != 0)
        {
            clock_t start = clock();
            if (des.rows != 0)
                hconcat(Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskBB(id),NBINS_EDDIR), des,des);
            else
                des=Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskBB(id),NBINS_EDDIR).clone();
            timeEDGESDIR += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
        }
        
        if (mCONEDDIR != 0)
        {
            clock_t start = clock();
            if (des.rows != 0)
                hconcat(Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskNeigboursBB(id),NBINS_EDDIR), des,des);
            else
                des=Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskNeigboursBB(id),NBINS_EDDIR).clone();
            timeEDGESDIR += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
        }
        
        //HOG
        if (mHOG != 0)
        {
            //get image BB superpixel
            Mat imageSP = cropSuperpixel(image,id,1,BB_SP_MASK).clone();
           
            if (des.rows != 0)
                hconcat(Descriptors::DescriptorHOG(imageSP), des,des);
            else
                des=Descriptors::DescriptorHOG(imageSP).clone();
            
        }
        if (mCONHOG != 0)
        {
            Mat imageSP = cropSuperpixel(image,id,1,BB_NEIG_MASK).clone();
            
            if (des.rows != 0)
                hconcat(Descriptors::DescriptorHOG(imageSP), des,des);
            else
                des=Descriptors::DescriptorHOG(imageSP).clone();
        }
        
        if (mSEMANTIC != 0)
        {
            if (des.rows != 0)
                hconcat(_arraySP[id].descriptorsSEMANTIC(SEMANTIC_LABELS),des,des);
            else
                des=_arraySP[id].descriptorsSEMANTIC(SEMANTIC_LABELS).clone();
            
        }
        
        if (mCONTEXT != 0)
        {
            if (des.rows != 0)
                hconcat(_arraySP[id].descriptorsCONTEXT(CONTEXT_LABELS),des,des);
            else
                des=_arraySP[id].descriptorsCONTEXT(CONTEXT_LABELS).clone();
            
        }
        
        if (mCONTEXT2 != 0)
        {
            if (des.rows != 0)
                hconcat(_arraySP[id].descriptorsCONTEXT_ORIENTED(CONTEXT2_LABELS),des,des);
            else
                des=_arraySP[id].descriptorsCONTEXT_ORIENTED(CONTEXT2_LABELS).clone();
                    
        }
        
        if (mGLOBAL != 0)
        {
            if (des.rows != 0)
                hconcat(_arraySP[id].descriptorsCONTEXT_GLOBAL(GLOBAL_LABELS),des,des);
            else
                des=_arraySP[id].descriptorsCONTEXT_GLOBAL(GLOBAL_LABELS).clone();
            
        }

        return des.clone();
    }
    
    /*************************************************************************************
     * initializeLabeling()
     *
     */
    Mat initializeLabeling(string path, int mode = MODE_LABEL_MEDIAN)
    {
        //read image
        try{
            _labelsInput = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
            
            
             printf("Image Labeling not zero %d\n",countNonZero(_labelsInput));
           
            
            _labelsInput = (_labelsInput * (NUMLABELS - 1)/ 255) ;
            
            if (_labelsInput.rows != _image.rows || _labelsInput.cols != _image.cols)
            {
                resize(_labelsInput, _labelsInput, Size(_image.cols,_image.rows));
            }

           /*double min, max;
            cv::minMaxLoc(_labelsInput, &min, &max);
            printf("%f %f ",min,max);//*/
                   
            if(_labelsInput.data == NULL)
            {
                printf("Image Labeling %s not found\n",path.c_str());
                return Mat::zeros(100, 100, CV_8UC1);
            }
            else
               if (_DEBUG == 1) printf("Mat _labels CV_8UC1 rows %d cols %d\n",_image.rows,_image.cols);
        }
        catch(int e)
        {
            if (_DEBUG == 1) printf("Image Labeling %s not found\n",path.c_str());
            return Mat::zeros(100, 100, CV_8UC1);
        }
        //Mat im;
        for (int id=0; id < maxID+1; id++)
        {
            
            int l = _arraySP[id].create_labelHist(_labelsInput,NUMLABELS,mode);
            _labels.setTo(l,_arraySP[id].getMask());
           // printf("id %d label: %d\n",id,l);
        }

        //paint
        labelSet val(NUMLABELS);
        Mat leyend= Mat::ones(_image.rows,_image.cols, CV_8UC3);
        
       
        
        return val.paintLabelRandom(_labels,NUMLABELS,&leyend);//out;

    }//initializLabeling
    
    Mat initializeSegmentation(string path, int numLabels, int mode = MODE_LABEL_NOTZERO, int labelSegmen = 60)
    {
        //read image
        Mat seg;
        try{
            seg =  imread(path,CV_LOAD_IMAGE_GRAYSCALE);
            
            
            if ( seg.rows != _image.rows && seg.cols != _image.cols)
            {
                resize(seg, seg, Size(_image.cols,_image.rows),0,0,INTER_NEAREST);
            }
            
            //if (labelSegmen == 60)
                seg = (seg * (labelSegmen)/ 255);
            //else
                
            
            //////////////////
           /* labelSet val(12);
            Mat l = Mat::zeros(seg.rows,seg.cols,CV_8UC3);
            
            Mat out = val.paintLabelRandom(seg, 12, &l).clone();
            imshow("input ",out);
            imshow("leyend",l);
            waitKey(0);//*/
            ///////////////////*/
            
            /*double min, max;
            cv::minMaxLoc(seg, &min, &max);
            printf("initialize segmentation %f %f\n",min,max);//*/
            
            if(seg.data == NULL)
            {
                printf("Image Segmentation %s not found\n",path.c_str());
                return Mat::zeros(100, 100, CV_8UC1);
            }
            else
                if (_DEBUG == 1) printf("Mat segmentation CV_8UC1 rows %d cols %d\n",_image.rows,_image.cols);
        }
        catch(int e)
        {
            printf("Image Segmentation %s not found\n",path.c_str());
            return Mat::zeros(100, 100, CV_8UC1);
        }
        
        Mat im = Mat::zeros(_image.rows,_image.cols,CV_8UC1);
        labelSet val(numLabels);

        if (numLabels == 23)
        {
            seg = val.convert2pascalcontextNoObjects(seg).clone();
        }
        
        for (int id=0; id < maxID+1; id++)
        {
            int l =  _arraySP[id].create_labelSegmentation(seg,numLabels,mode,getMaskBB(id));
            im.setTo(l,_arraySP[id].getMask());
            //printf("LABEL: %d %s \n",l,val.getLabel(l).c_str());getchar();
        }
        
        //////////////////
        /*labelSet val1(60);
        Mat l = Mat::zeros(im.rows,im.cols,CV_8UC3);
        
        Mat out2 = val1.paintLabelRandom(im, 60, &l).clone();
        imshow("INITIAL SEGMENTATION",out2);
        imshow("leyend",l);*/
        ///////////////////
        //
        
        calculateLabelingNeighbours(seg,numLabels);
        
        //paint
        Mat leyend= Mat::ones(_image.rows,_image.cols, CV_8UC3);
       
        //leyend.release();
        return val.paintLabelRandom(seg,numLabels,&leyend);
    }
    
    void calculateLabelingNeighbours(Mat seg,int numLabels)
    {
        clock_t start = clock();
        
        //create image new
        Mat newLabels = Mat::zeros(_image.rows,_image.cols, CV_8UC1);
        
        for (int id1=0; id1 < maxID+1; id1++)
        {
            //get neighbour
            set<int> neig = _arraySP[id1].getFirstNeighbours();
            int ln=0;
            
            std::set<int>::iterator it;

            Mat maskN = Mat::zeros(_image.rows,_image.cols, CV_8UC1);
            for (it=neig.begin(); it!=neig.end(); ++it)
            {
                ln = _arraySP[id1].addHistogramLabelSegmentation(_arraySP[*it].getLabelSegmentation());
                //printf("id: %d neig: %d l=%d  %d\n",id1,*it,ln,(int)neig.size());//getchar();
                //concat mask
                bitwise_or(maskN, _arraySP[*it].getMask(),maskN);
            }
            
            _arraySP[id1].normalizeLabelFirstSegmentation((int)neig.size());
            newLabels.setTo(ln,_arraySP[id1].getMask());
            
            //create oriented semantic label segmentation
            _arraySP[id1].create_labelOriented(seg,numLabels,getMaskNeigboursBB(id1));
            _arraySP[id1].create_labelOrientedGlobal(seg,numLabels);

        }
        
        /*labelSet val(60);
        Mat l;// = Mat::zeros(seg.rows,seg.cols,CV_8UC3);
        val._DEBUG=1;
        Mat out = val.paintLabelRandom(newLabels, 60, &l).clone();
        imshow("NEIG SEGMENTATION",out);*/
        
        
        timeSEMANTIC += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);///(float) maxID;
    }//calculateLabelingNeighbour*/
    

};

#endif // SUPERPIXELS_H
