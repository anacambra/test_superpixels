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
#include "labelSet.cpp"

//vlfeat
#include "vl/generic.h"
#include "vl/slic.h"

#include <opencv2/ml/ml.hpp>


using namespace std;

class SuperPixels
{
 
    Mat _image; // original image in color BGR
    Mat _ids; // superpixels ids  CV_8UC1
    Mat _sobel; //MASK superpixel boundaries  UCHAR
    Mat _labelsInput; //input labeling
    
    Mat _labels; // labeling CV_32FC1
    
    // superpixels params
    int _TAM_SP = 60;
    int _NUM_MAX_SP = 700;
    
    SuperPixel *_arraySP;
   
    int NUMLABELS=60;
    
    unsigned char _DEBUG = 1;

    
public:
    
    int maxID;
    
    SuperPixels(){ maxID=0; }
    ~SuperPixels(){ _image.release(); _ids.release(); _sobel.release(); }
    
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
                printf("Mat _image CV_8UC1 rows %d cols %d\n",_image.rows,_image.cols);
            
            _ids= Mat::zeros(_image.rows,_image.cols,CV_8UC1);
            _sobel= Mat::zeros(_image.rows,_image.cols,CV_8UC1);
            _labels= Mat::ones(_ids.rows,_ids.cols,CV_32FC1)*-1;
        }
        catch(int e)
        {
            printf("Image %s not found\n",path.c_str());
            return;
        }
        
        size_t found = path.find_last_of(".");
        string name = path.substr(0,found) + "_" + to_string(_TAM_SP)+".sp";
        FILE *f = fopen(name.c_str(),"r");
        
        if (f!=NULL)
        {
            fclose(f);
            
            if (_DEBUG == 1) start = clock();
            
            loadSuperPixels(name);
            
            if (_DEBUG == 1) printf("**** TIME: load Superpixels: %f seconds\n ",(float) (((double)(clock() - start)) / CLOCKS_PER_SEC) );
        }
        
        else
        {
            if (_DEBUG == 1) start = clock();
            
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
        return im;
    }
    
    Mat paintSuperpixel(Mat image, int id){
        
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
        
        return im;
    }//paintSuperpixel
    
    /*************************************************************************************
     * calculateSLICSuperpixels
     *
     *      mat: BGR 8 bit channels
     *      fill : _ids, maxID
     *      limit total number of superpixels (maxID+1)
     */
    void calculateSLICSuperpixels(Mat mat){
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
        vl_size minRegion = region - 5;
        
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
                _ids.at<uchar>(i,j)=label;
                
                
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
                    int id = _ids.at<uchar>(i,j);
                    fwrite(&id, sizeof id, 1, f);
                }
            
            fclose(f);
            
        }catch(int e){
            printf("Exception!");}
        
    }//superpixels2file
    
    
    /*************************************************************************************
     * loadSLICSuperpixels
     *
     */
    void loadSuperPixels(string path)
    { //printf("To-Do: not implemented yet");}//loadSuperPixels
        FILE *f;
        int w=0,h=0;
        maxID = 0;
        
        try{
            f = fopen(path.c_str(),"rb");
            h=_image.rows; w=_image.cols;
             _ids= Mat::zeros(h,w,CV_8UC1);
            
            for(int i=0;i<h;i++)
                for (int j=0; j<w; j++)
                    if(!feof(f))
                    {
                        int id;
                        fread(&id,sizeof(int),1,f);
                        if (id >= maxID) maxID = id;
                        _ids.at<uchar>(i,j)=id;
                        //printf("%d %d %d\n",i,j,(int)_ids.at<float>(i,j));
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
            }
        }
        else{
            printf("_ids superpixels NULL \n");
            return;
        }
        
        // NEIGHBOUGRS
        
        //first boundaries in _sobel
        for (int x = 0; x < _sobel.rows; x++)
        {
            for (int y = 0; y < _sobel.cols; y++)
            {
                if ( _sobel.at<uchar>(x,y) == 255) //boundarie
                {
                    int id = (int)_ids.at<uchar>(x,y);
                    
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
                                int v = (int)_ids.at<uchar>(i,j);
                                _arraySP[id].addFirstNeighbour(v);
                                
                                
                            }//if neighbours
                        }//for j
                    }//for i
                }// if boundarie
            }//for y
        }//for x
        
        //second boundaries
        for (int id=0; id < maxID+1; id++)
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
    
    void calculateDescriptors()
    {
        
        int nSAMPLES = 4;
        //int numDesc = 100 + 255 + 255;
        Mat trainingData = Mat::zeros(nSAMPLES,100,CV_32FC1);
        
        Mat l0 = Mat::zeros(1,100,CV_32FC1);
        _arraySP[2].descriptors(_image,&l0);
        
        //l0.row(0).copyTo(trainingData.row(0));
        l0.row(0).copyTo(trainingData.row(0));
        
      //  for( int h = 0; h < 100; h++ )
            // printf("%d ", trainingData.at<uchar>(0,h));
        
        
       // Mat l1 = Mat::zeros(1,100,CV_8UC1);
        _arraySP[1].descriptors(_image,&l0);
        
        //l0.row(0).copyTo(trainingData.row(0));
        l0.row(0).copyTo(trainingData.row(1));
        
        //for( int h = 0; h < 100; h++ )
          //  printf("%d ", trainingData.at<uchar>(1,h));
        
        _arraySP[0].descriptors(_image,&l0);
        
        //l0.row(0).copyTo(trainingData.row(0));
        l0.row(0).copyTo(trainingData.row(2));
        
        _arraySP[8].descriptors(_image,&l0);
        //l0.row(0).copyTo(trainingData.row(0));
        l0.row(0).copyTo(trainingData.row(3));
        //for( int h = 0; h < 100; h++ )
        //printf("%d ", trainingData.at<uchar>(2,h));
        
       
 
        // Set up training data
        float labels[4] = {0.0, 0.0, 1.0, 1.0};
        Mat labelsMat(nSAMPLES, 1, CV_32FC1, labels);
        
       // float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
        Mat trainingDataMat(nSAMPLES, 100, CV_32FC1, &trainingData);
        
        // Set up SVM's parameters
        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
        
        // Train the SVM
        CvSVM SVM;
        SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
        
        //SVM.save("svmTEXT.xml");
       // SVM.load("svmTEXT.xml");
        //test
        
        Mat l1 = Mat::zeros(1,100,CV_32FC1);
        _arraySP[2].descriptors(_image,&l1);
        
        float response = SVM.predict(l1);
        printf("RESPONSE 2  %f\n",response);waitKey(0);
        
        _arraySP[1].descriptors(_image,&l1);
        response = SVM.predict(l1);
        printf("RESPONSE 1  %f\n",response);waitKey(0);
        
        _arraySP[8].descriptors(_image,&l1);
        response = SVM.predict(l1);
        printf("RESPONSE 8  %f\n",response);waitKey(0);
        
        _arraySP[0].descriptors(_image,&l1);
        response = SVM.predict(l1);
        printf("RESPONSE 0  %f\n",response);waitKey(0);
        
        waitKey(0);
        
    }
    
    /*************************************************************************************
     * initializeLabeling()
     *
     */
    Mat initializeMeanLabeling(string path)
    {
        //read image
        try{
            _labelsInput = imread(path,CV_LOAD_IMAGE_UNCHANGED);
            _labelsInput = (_labelsInput / 255)*NUMLABELS;
            
            
           /* double min, max;
            cv::minMaxLoc(_labels, &min, &max);
            printf("%f %f ",min,max);//*/
                   
            if(_labelsInput.data == NULL)
            {
                printf("Image Labeling %s not found\n",path.c_str());
                return Mat::zeros(100, 100, CV_8UC1);
            }
            else
                printf("Mat _labels CV_8UC1 rows %d cols %d\n",_image.rows,_image.cols);
        }
        catch(int e)
        {
            printf("Image Labeling %s not found\n",path.c_str());
            return Mat::zeros(100, 100, CV_8UC1);
        }
        
        for (int id=0; id < maxID+1; id++)
        {
            int l=_arraySP[id].create_labelHist(_labelsInput,NUMLABELS);
            _labels.setTo(l,_arraySP[id].getMask());
            
           /* imshow("HIST labels",_arraySP[id].paintHistogram());
            waitKey(0);//*/
        }
        
        //paint
        labelSet val(NUMLABELS);
        Mat leyend= Mat::ones(_image.rows,_image.cols, CV_8UC3);
        Mat out = val.paintLabelRandom(_labels,NUMLABELS,&leyend).clone();//
        
        return out;


    }//initializLabeling
    

};

#endif // SUPERPIXELS_H
