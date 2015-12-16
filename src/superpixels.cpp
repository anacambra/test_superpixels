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

//vlfeat
#include "vl/generic.h"
#include "vl/slic.h"

//sistema de ecuaciones
#include <cmath>
#include <vector>
#include <iomanip>
using namespace std;
/*#include "least-squares-linear-system2.h"
using namespace Eigen;

typedef Matrix<double, Dynamic, 1> MatrixFX;*/

/*#define RED 0
#define GREEN 1
#define RANDOM 2


// mode solve sistem
#define CTE 0
#define LINEAL 1
#define MULTI 2

#define NUMLABELS 20


/*#define DEPTH_NEAR 224.0
#define DEPTH_FAR_1 160.0
#define DEPTH_FAR_2 96.0
#define DEPTH_FAR_3 32.0*/

/*#define DEPTH_NEAR 229
#define DEPTH_FAR_0 179
#define DEPTH_FAR_1 128
#define DEPTH_FAR_2 77
#define DEPTH_FAR_3 26*/

/*#define DEPTH_NEAR 130//173//229.5
 #define DEPTH_FAR_0 86//125//178.5
 #define DEPTH_FAR_1 75//116//127.5
 #define DEPTH_FAR_2 33//69//76.6
 #define DEPTH_FAR_3 18//47//25.5*/

/*#define DEPTH_NEAR 217
 #define DEPTH_FAR_0 160
 #define DEPTH_FAR_1 109
 #define DEPTH_FAR_2 77
 #define DEPTH_FAR_3 63*/

/*#define DEPTH_NEAR 130
 #define DEPTH_FAR_0 105
 #define DEPTH_FAR_1 80
 #define DEPTH_FAR_2 46
 #define DEPTH_FAR_3 25*/


class SuperPixels
{
 
    Mat _image; // original image in color BGR
    Mat _ids; // superpixels ids CV_32FC1
    Mat _sobel; //boundaries superpixels uchar!
    
    Mat _labels; // labeling CV_32FC1
    
    // superpixels params
    int _TAM_SP = 40;
    int _NUM_MAX_SP = 700;
    
    SuperPixel *_arraySP;
   
    
    unsigned char _DEBUG = 1;
    
    /*****************************************************************/
    
    
    
    
    /*****************************************************************/
    
public:
    
    int maxID;
    
    SuperPixels(){ maxID=0; }
    ~SuperPixels(){ _image.release(); _ids.release(); _sobel.release(); }
    
    /*************************************************************************************
     * SuperPixels
     *  load _image and obtain superpixels
     *  superpixel boundaries
     *  initizalize superpixels
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
            
            _ids= Mat::zeros(_image.rows,_image.cols,CV_32FC1);
            _sobel= Mat::zeros(_image.rows,_image.cols,CV_32FC1);
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
        
        //init superpixels
        calculateBoundariesSuperpixels();
        initializeSuperpixels();
        return;
    }//SuperPixels

    
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
                    int id = (int)_ids.at<float>(i,j);
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
             _ids= Mat::zeros(h,w,CV_32FC1);
            
            for(int i=0;i<h;i++)
                for (int j=0; j<w; j++)
                    if(!feof(f))
                    {
                        int id;
                        fread(&id,sizeof(int),1,f);
                        if (id >= maxID) maxID = id;
                        _ids.at<float>(i,j)=(float)id;
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
                mask_sp = (_ids == (float)i);
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
    
    
   // void infoSuperpixels2file(std::string nameFile);
   
    /* Mat _depth;
    Mat _ids;
    Mat _imDepth;
    Mat _pixelDepth;
    
    Mat _lab;
    
    Mat _prob[21];
    
    int showInfo;
    
    int num_eqU=0;//num unary equations
    int num_eqBcolor=0;
    int num_eqB=0;
    
    int _TAM_SP = 20;//30;//30;//35;//35;//25;
    double w_unary=0.5;
    double w_color=0.9;
    
    double _MAX_DIFF_LAB = 60;//15.0;
    
    int _NUM_MAX_SP = 700;//600;
   
    int MAX_BINARIAS=5000000;*/
    
    ////// Superpixels funciones
  /*  int numCoef=3;
    
    Optimization::LeastSquaresLinearSystem<double> ecuaciones;
    Optimization::LeastSquaresLinearSystem<double> unary;
    Optimization::LeastSquaresLinearSystem<double> unaryCTE;
    
    Optimization::LeastSquaresLinearSystem<double> binary;
    Optimization::LeastSquaresLinearSystem<double> binaryCOLOR;
    Optimization::LeastSquaresLinearSystem<double> binaryGRADIENT;//
    
    Optimization::LeastSquaresLinearSystem<double> unaryMulti[21];
    Optimization::LeastSquaresLinearSystem<double> unaryMultiUser[21];
    
    float timeU = 0.0;
    float timeB = 0.0;
    float timeSolve = 0.0;*/
    
    
//public:
   /* SuperPixel *arraySP;
    Mat _sobel;
   // Mat _pixelDepth;
    int maxID;
    SuperPixels(std::string path);
    ~SuperPixels() ;

    //QImage point to the data of _image
   // QImage getImage();
    Mat getImage();
    Mat imageDepth();
    Mat getMask(int id);
    void updateDepth(Mat depth);
    
    //QImage mat_to_qimage_ref(Mat &mat, QImage::Format format);
    void loadSuperPixels(std::string path);
    void calculateSLICSuperpixels(Mat mat);
    void loadDepth(std::string path);
    void loadDepth(Mat input);
    void loadUserDepth(std::string path);
    void loadDepthBuildSystem(std::string path);
    void loadDepthBuildSystemCoef();
    void loadDepthBuildSystemCoef(std::string path);

    Mat paintSuperpixel(int i,int y);
    Mat paintSuperpixel(int i,int y,Scalar *color);
    void paintSuperpixelByID(int id,int grayLevel);
    void copyDepth(int x, int y, float depth);
    void copySuperpixel(int x,int y, int last_x,int last_y);
    
    void calcularVecinos();
   // void calcularVecinos2();
    void calcularVecinos2(int id, int *array);

    int getIdFromPixel(int x, int y);
    int getLabel
    /*float getMedianDepth(int x, int y);
    float getDepthFromPixel(int x, int y);
    float getDepthInitialFromPixel(int x, int y);
    bool isNotNullImage();
    bool isNotNullDepth();
    bool isNotNullIndex();

    void resetImage();
    void paintZeroDepth();
    void depthWithBorders();*/
    
   // void infoSuperpixels2file(std::string nameFile);

   // float medianHistogram(MatND hist,int numPixels);
    
  //  MatND calHistogram(Mat m, Mat mask);
   // void histogramLAB(int id,Mat mask);
  /*  float cmpDepth(int a, int b, int mode);
    float cmpLab(int a, int b, int mode);
    float cmpLab(int a, SuperPixel b, int mode);
    
    float getDepth(int i);
    float getVar(int i);
    float getAccu(int i);
    
    //equations
    void buildEquationSystem();
    void addEquationsBinaries();
    void addEquationsBinariesCoef();
    void addEquationsBinariesBoundaries(bool verbose);
    void addEquationsBinariesBoundaries2();
    void addEquationsBinariesBoundariesCoef(bool verbose);
    void addEquationsBinariesBoundariesCoef();
    void addEquationsBinariesGradientCoef();
    float similarityBinariesBoundaries(int s1, int s2);
    void addEquationsUnaries(int id,float di);
    void addEquationsUnaries(int x,int y,float di);
    void addEquationsUnaries();
    void addEquationsUnariesMulti(int l);
    void addUnariesMultiUser(int l, int x,int y);
    void addUnariesCoef(int x, int y, double di);
    void addEquationsUnariesMedianCoef();
    void addEquationsUnariesCoef();
    void solve();
    void solveCoef();
    Mat solveMulti();
    
    void resetDepthEquations();
    void resetDepthEquationsCoef();
    
    //effects
    Mat blurImage(Mat image, Mat imageDepth, int nbins);
    Mat blurImage(Mat image, Mat imageDepth, int nbins, double minFocus, double maxFocus,double size);
    //code adolfo
    Mat blurImageDepth(const cv::Mat& image, const cv::Mat& depth,
                         int nbins, float focal_distance, float focal_length, float aperture, bool linear);*/

};

#endif // SUPERPIXELS_H
