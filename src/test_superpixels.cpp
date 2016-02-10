//
//  main.cpp
//  test_superpixels
//
//  Created by Ana Cambra on 26/11/14.
//  Copyright (c) 2014 Ana Cambra. All rights reserved.
//

#include <iostream>

#include "superpixels.cpp"
#include <time.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>
using namespace cv;

using namespace boost::filesystem;


//#include <caffe/caffe.hpp>
//using namespace caffe;


#define LABEL_TEXT 1
#define LABEL_NOTEXT -1

//#define DEBUG_SP 1

int mLAB   = 0; int NBINS_L = 50; int NBINS_AB=128;
int mRGB   = 0; int NBINS_RGB   = 256;
int mPEAKS = 0; int NBINS_PEAKS = 128;

int mLINES = 0; int NBINS_LINES = 128;

int mEDGES = 0; int NBINS_EDGES; int modeEDGES = 1;

int mCAFFE = 0; string CAFFE_LAYER = "fc7"; int NUMCAFFE = 4096;

int mSEMANTIC = 0; int SEMANTIC_LABELS = 60;

/*int nSAMPLES = 50;
string nameSVM = "svm_LAB_PEAKS_" + to_string(nSAMPLES) + "_NOTZERO.xml";*/

//train SVM
void descriptorSVMText(string dir_path,string dir_pathGT, string dir_edges, int nSAMPLES, string nameSVM, string out);

//HOG
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size );
void compute_hog( Mat  img_lst, vector< Mat > & gradient_lst, const Size & size );


int main(int argc, const char * argv[]) {

    
    //parse argv
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("v", "debug mode ON")
    ("svmOptions", boost::program_options::value<std::string>()->required(), "descriptors train")
    ("svmTest", "test svm")
    ("image", boost::program_options::value<std::string>(), "image")
    ("labeled", boost::program_options::value<std::string>(), "image GT labeled")
    ("edges", boost::program_options::value<std::string>(), "path image edges")
    ("numLabels", boost::program_options::value<int>()->default_value(2), "num labels")
    ("svmTrain", "train svm")
    ("nSamples", boost::program_options::value<int>()->required(), "test svm");
    
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv,desc),parameters);
    boost::program_options::notify(parameters);
    
    
    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    
    bool DEBUG = false;
    if (parameters.find("v") != parameters.end()) {
        DEBUG = true;
    }
    
    //SVM
    int nSAMPLES = parameters["nSamples"].as<int>();
    string nameSVM = "svm_" + parameters["svmOptions"].as<std::string>() + "_" + to_string(nSAMPLES) + "_NOTZERO.xml";
    
    if (parameters["svmOptions"].as<std::string>().find("LAB") != std::string::npos)
        mLAB = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("RGB") != std::string::npos)
        mRGB = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("PEAKS") != std::string::npos)
        mPEAKS = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("EDGES") != std::string::npos)
    {
        mEDGES = 1;
        
        if (modeEDGES == 0) //hist
            NBINS_EDGES = 100;
        else if (modeEDGES == 1) //moments
            NBINS_EDGES = 4;
        else if (modeEDGES == 2)
            NBINS_EDGES = 100 + 4;
        
    }
    
    if (parameters["svmOptions"].as<std::string>().find("CAFFE") != std::string::npos)
        mCAFFE = 1;
    if (parameters["svmOptions"].as<std::string>().find("SEMANTIC") != std::string::npos)
        mSEMANTIC = 1;
    
    string nameSVM2 = "train/" + nameSVM;

    
    ////////////////////////
    //SVM train o load
   
    if (parameters.find("svmTrain") != parameters.end()) {
        
        printf("==================================================\n");
        printf("**** \tTRAIN SVM_%s numSamples %d\n",parameters["svmOptions"].as<std::string>().c_str(),nSAMPLES);
        printf("==================================================\n");
        
        ifstream infile(nameSVM2);
        if (! infile.good()) {
            
            descriptorSVMText("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images",
                              "/Users/acambra/Dropbox/dataset/ICDAR/masks",
                              "/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images/edges",
                              nSAMPLES,
                              nameSVM2,
                              "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/train/des_"+nameSVM+".yaml");
        }
        else{
            printf("\t* SVM Load: %s\n", nameSVM2.c_str());
        }
        
    }//parameters.find("svmTrain")
    
    ////////////////////////
    //evaluate test SVM
    
    if (parameters.find("svmTest") != parameters.end()) {
        
        CvSVM SVM;
        SVM.load(nameSVM2.c_str());
        
        boost::filesystem::path inputImage(parameters["image"].as<std::string>());
        
        ////////////////////////
        //SUPERPIXELS
        
        SuperPixels *SPCTE;
        string nameImage = inputImage.string();//argv[1];
        SPCTE = new SuperPixels(nameImage);
        
        //boundaries between SP
        SPCTE->calculateBoundariesSuperpixels();
        
        //init superpixels
        SPCTE->initializeSuperpixels();
        Mat imgSP = SPCTE->getImageSuperpixels().clone();
        //imshow("superpixels",SPCTE->getImageSuperpixels());
        // waitKey(0);//*/
        
        if (mCAFFE == 1) SPCTE->initCaffe();
        
        //check neigbour
        // Mat image
        /*for (int id=0; id < SPCTE->maxID+1; id++)
         {
         imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
         waitKey(25);
         }*/
        
        //TEXT LABELS
        SPCTE->setNUMLABELS(parameters["numLabels"].as<int>());//atoi(argv[3]));
        //init labeling
        Mat gt = SPCTE->initializeLabeling(parameters["labeled"].as<std::string>(), MODE_LABEL_NOTZERO).clone();
        imshow("GT", gt);
        char k=-1;
        
        string nameWindow = "Superpixel ";
        
        //test
        for (int id=0; id < SPCTE->maxID+1; id++)
        {
            if (DEBUG == 1) {
                //KEYBOARD
                if (k == 2){//<-
                    if (id >= 2) id = id - 2;
                    if (id == 1) id = 0;
                }
                if (k == 27){//ESC
                    break;
                }
            }
            
            destroyWindow(nameWindow.c_str());
            nameWindow = "Superpixel " + to_string(id);
            
            //
            imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
            Mat img_sp;
            Mat imgEdges = imread(parameters["edges"].as<std::string>(),CV_LOAD_IMAGE_COLOR);
            //imshow("img EDGES ",imgEdges);
            
            Mat desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                    ,mLAB,NBINS_L,NBINS_AB \
                                                    ,mRGB,NBINS_RGB\
                                                    ,mPEAKS,NBINS_PEAKS \
                                                    ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                    ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                                                    ,mSEMANTIC,SEMANTIC_LABELS).clone();
            
            //evaluate
             float response = SVM.predict(desID);//,true);
             printf("RESPONSE SVM id: %d  %f\n",id,response);
             
             //paint image with SVM response
             if (response >= LABEL_TEXT)
             imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();//*/
            
            //
            
           //TEST SEMANTICC
            
           
            Mat seg = SPCTE->initializeSegmentation("/Users/acambra/Dropbox/pascalcontext/ICDAR-fcn8/img_1.png",60);
            
            
           /* imshow("SEGMENTATION", seg);
            waitKey(0);*/
            
            if (DEBUG == 1) {
                
                imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
                Mat img_sp;
                Mat imgEdges = imread(parameters["edges"].as<std::string>(),CV_LOAD_IMAGE_COLOR);
                //imshow("img EDGES ",imgEdges);
                
                Mat desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                        ,mLAB,NBINS_L,NBINS_AB \
                                                        ,mRGB,NBINS_RGB\
                                                        ,mPEAKS,NBINS_PEAKS \
                                                        ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                        ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                                                        ,mSEMANTIC,SEMANTIC_LABELS).clone();
                
                
                
                //evaluate
               /* float response = SVM.predict(desID);//,true);
                printf("RESPONSE SVM id: %d  %f\n",id,response);
                
                //paint image with SVM response
                if (response >= LABEL_TEXT)
                    imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();//*/
                
                if ((char)k != 'c')
                    k = waitKey(0);

            
            }
            
            
            
        }//*/
        
        if (DEBUG == 1){
            imshow(nameSVM,imgSP);waitKey(0);}
        else{
            
            size_t found1 = nameImage.find_last_of("/");
            string name = "train/test/"+ nameImage.substr(found1+1) + "_" +parameters["svmOptions"].as<std::string>()+ "_" + to_string(nSAMPLES)+ ".png";
            imwrite(name,imgSP);
        }
    }//svmTest
    
    
    
    //check neigbour
   // Mat image
    /*for (int id=0; id < SPCTE->maxID+1; id++)
    {
        imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
        waitKey(25);
    }*/
    
    //init labeling
    //Mat out = SPCTE->initializeMeanLabeling(argv[2]).clone();
    //imshow("labels", out);
    
    return 0;
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TRAIN SVM

void descriptorSVMText(string dir_path,string dir_pathGT, string dir_edges, int nSAMPLES, string nameSVM, string out)
{
    
    //SVM
    int numDES = 0;
    if (mLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
    if (mRGB == 1)      numDES += (3*NBINS_RGB);
    if (mPEAKS == 1)    numDES += NBINS_PEAKS;
    if (mEDGES == 1)    numDES += NBINS_EDGES;
    if (mCAFFE == 1)    numDES += NUMCAFFE;
    
    
    Mat trainingData = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    Mat labels = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
    
    int pos=0;
    int neg=0;
    
    
    for (auto i = directory_iterator(dir_path); i != directory_iterator() && (pos+neg < nSAMPLES*2); i++)
    {
        if (!is_directory(i->path()))
        {
            if ((pos+neg >= nSAMPLES*2))
                break;
            
            string nameImage = dir_path + "/" + i->path().filename().string();
            string extension = i->path().filename().extension().string();
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                printf("Image (%d): %s\n",(pos + neg),i->path().filename().string().c_str());
                SuperPixels *SPCTE;
                SPCTE = new SuperPixels(nameImage);
                //boundaries between SP
                SPCTE->calculateBoundariesSuperpixels();
                //init superpixels
                SPCTE->initializeSuperpixels();
                //TEXT LABELS
                SPCTE->setNUMLABELS(2);//atoi(2));
                //SEMANTIC SEGMENTATION
                string imageGT = dir_pathGT + "/gt_" + i->path().filename().string();
                size_t lastindex = imageGT.find_last_of(".");
                imageGT = imageGT.substr(0, lastindex) + ".png";
                
                SPCTE->initializeLabeling(imageGT, MODE_LABEL_NOTZERO);//MODE_LABEL_MEDIAN);//
                
                string nameEdges = dir_edges + "/" + i->path().filename().string();
                nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                Mat imgEdges = imread(nameEdges,CV_LOAD_IMAGE_COLOR);
                
                for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                {
                    
                    Mat des = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                          ,mLAB,NBINS_L,NBINS_AB \
                                                          ,mRGB,NBINS_RGB\
                                                          ,mPEAKS,NBINS_PEAKS \
                                                          ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                          ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                                                          ,mSEMANTIC,SEMANTIC_LABELS).clone();
                    //add des in trainingData
                    //des.row(0).copyTo(trainingData.row(id));
                    int n = neg + pos;
                    
                    if ((SPCTE->_arraySP[id].accLabel(1) == 1.0) && (pos < nSAMPLES))
                    { //text
                        labels.at<float>(n,0) = (float) LABEL_TEXT;
                        des.row(0).copyTo(trainingData.row(n));
                        pos = pos + 1;
                    }
                    else
                    {
                        if ((SPCTE->_arraySP[id].accLabel(0) == 1.0) && (neg < nSAMPLES))
                        {
                            labels.at<float>(n,0) = (float) LABEL_NOTEXT;
                            //add des in trainingData
                            des.row(0).copyTo(trainingData.row(n));
                            neg = neg + 1;
                        }
                    }
                    des.release();
                }//for superpixels
                delete(SPCTE);
            }//if
        }
    }//for
    
    //SAVE TRAIN DESCRIPTORS
    // string filename = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/svm_train.yaml";
    FileStorage fs(out, FileStorage::WRITE);
    fs << "trainingData" << trainingData;
    fs << "labels" << labels;
    fs.release();
    //cout << trainingData << endl;
    
    //TRAIN
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
    
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingData, labels, Mat(), Mat(), params);
    
    SVM.save(nameSVM.c_str());
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*   HOG */

//void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size )
void compute_hog(  Mat  img, vector< Mat > & gradient_lst, const Size & size )
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
    
   // vector< Mat >::const_iterator img = img_lst.begin();
   // vector< Mat >::const_iterator end = img_lst.end();
   // for( ; img != end ; ++img )
   // {
        cvtColor( img, gray, COLOR_BGR2GRAY );

        hog.compute( gray, descriptors, hog.blockStride, Size( 0, 0 ), location );
    
    
    
        gradient_lst.push_back( Mat( descriptors ).clone() );
//#ifdef _DEBUG
        imshow( "gradient", get_hogdescriptor_visu( img.clone(), descriptors, size ) );
       // waitKey( 0 );
//#endif
    //}
}

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 1;
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