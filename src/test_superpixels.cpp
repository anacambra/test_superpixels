//
//  main.cpp
//  test_superpixels
//
//  Created by Ana Cambra on 26/11/14.
//  Copyright (c) 2014 Ana Cambra. All rights reserved.
//

#include <iostream>

#include "superpixels.cpp"
//#include <time.h>

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

bool DEBUG = false;

//caffe
utilsCaffe *_caffe;

/*int nSAMPLES = 50;
string nameSVM = "svm_LAB_PEAKS_" + to_string(nSAMPLES) + "_NOTZERO.xml";*/

//train SVM
Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges = "", string nameSegmen = "");

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges , string dir_segmen , int nSAMPLES, string nameSVM, string out);


//initialize structure of Superpixels
SuperPixels* svmSuperpixelsTEXT( string nameImage, int numLabels = 2, string imageGT = "", string nameSegmen = "")
{
    
    SuperPixels *SPCTE = new SuperPixels(nameImage);
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    //init superpixels
    SPCTE->initializeSuperpixels();
    //TEXT LABELS
    SPCTE->setNUMLABELS(2);
    
    if (!imageGT.empty())
        SPCTE->initializeLabeling(imageGT, MODE_LABEL_NOTZERO);//MODE_LABEL_MEDIAN);//
    
   // if (mCAFFE == 1) SPCTE->initCaffe("/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel","/Users/acambra/Dropbox/test_caffe/deploy.prototxt");
    
    SPCTE->setNUMLABELS(numLabels);
    
    if (!nameSegmen.empty())
        SPCTE->initializeSegmentation(nameSegmen,SEMANTIC_LABELS);
    
    return SPCTE;

}

//Mat A = loadMatFromYML("Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images/normals/img_1.yml","N") ;
Mat loadMatFromYML(string  file, string  variable)
{
    Mat A;
    FileStorage fs(file, FileStorage::READ);
    fs[variable] >> A;
    fs.release();
    
    return  A;
}

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
    ("semantic", boost::program_options::value<std::string>(), "path image semantic segmentation")
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
    
    
    if (parameters.find("v") != parameters.end()) {
        DEBUG = true;
    }
    
    //SVM Options
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
    
    if (parameters["svmOptions"].as<std::string>().find("SEMANTIC") != std::string::npos)
        mSEMANTIC = 1;
    
    string nameSVM2 = "train/" + nameSVM;

    
    ////////////////////////
    //SVM train o load file
   
    if (parameters.find("svmTrain") != parameters.end()) {
        
        printf("==================================================\n");
        printf("**** \tTRAIN SVM_%s numSamples %d\n",parameters["svmOptions"].as<std::string>().c_str(),nSAMPLES);
        printf("==================================================\n");
        
        ifstream infile(nameSVM2);
        if (! infile.good()) {
            
            if (parameters["svmOptions"].as<std::string>().find("CAFFE") != std::string::npos)
            {
                mCAFFE = 1;
                //initCaffe
                string model = "/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel";
                string proto = "/Users/acambra/Dropbox/test_caffe/deploy.prototxt" ;
                //printf("Model CAFFE: %s PROTO: %s\n",model.c_str(),proto.c_str());getchar();
                
                _caffe = new utilsCaffe(model,proto);
            }
            
            trainSVMText("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images",
                              "/Users/acambra/Dropbox/dataset/ICDAR/masks",
                              "/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images/edges",
                              "/Users/acambra/Dropbox/pascalcontext/ICDAR-fcn8",
                              nSAMPLES,
                              nameSVM2,
                             "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/train/des_"+ nameSVM +".yaml");
            if (_caffe) delete _caffe;
            
            printf("\n\t* SVM save: %s\n\n", nameSVM2.c_str());

        }
        else{
            printf("\n\t* SVM Load: %s\n\n", nameSVM2.c_str());
        }
        
    }//parameters.find("svmTrain")*/
    
    ////////////////////////
    //evaluate test SVM
    
    if (parameters.find("svmTest") != parameters.end()) {
        
        CvSVM SVM;
        SVM.load(nameSVM2.c_str());
        
        boost::filesystem::path inputImage(parameters["image"].as<std::string>());
        
        ////////////////////////
        //SUPERPIXELS
        
        
        string nameImage = inputImage.string();
        string nameGT = parameters["labeled"].as<std::string>();
        string nameSegmen ="";
        if (mSEMANTIC == 1)
            nameSegmen = parameters["semantic"].as<std::string>();
       
        SuperPixels *SPCTE;
        
        SPCTE= svmSuperpixelsTEXT(nameImage,2,nameGT,nameSegmen);
      
        Mat imgSP = SPCTE->getImageSuperpixels().clone();
        /*imshow("superpixels",SPCTE->getImageSuperpixels());
        waitKey(0);//*/
        
        //if (mCAFFE == 1) SPCTE->initCaffe();
        
        //check neigbour
        // Mat image
        /*for (int id=0; id < SPCTE->maxID+1; id++)
         {
         imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
         waitKey(25);
         }*/
        

        char k=-1;
        
        string nameWindow = "Superpixel ";
        
        //test
        
        clock_t start = clock();

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
            
            //show superpixel
            
            /*destroyWindow(nameWindow.c_str());
            nameWindow = "Superpixel " + to_string(id);
            imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));*/
            
            string nameEdges = parameters["edges"].as<std::string>();
            
            string nameSegmen="";
            if (parameters.find("semantic") != parameters.end())
                 nameSegmen = parameters["semantic"].as<std::string>();
            
            //calculate descriptor superpixel
            
            
            Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
            
            
            //evaluate SVM
            float response = SVM.predict(desID);//,true);
            printf("RESPONSE SVM id: %d  %f\n",id,response);
            
            //paint image with SVM response
            if (response >= LABEL_TEXT)
                imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();//*/
            
            if (DEBUG == 1) {
                
                destroyWindow(nameWindow.c_str());
                nameWindow = "Superpixel " + to_string(id);
                imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
                
                
                
                
               /* imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
                Mat img_sp;
                Mat imgEdges = imread(parameters["edges"].as<std::string>(),CV_LOAD_IMAGE_COLOR);
                //imshow("img EDGES ",imgEdges);
                
                Mat desID = descriptorText(SPCTE,id);//= SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                        ,mLAB,NBINS_L,NBINS_AB \
                                                        ,mRGB,NBINS_RGB\
                                                        ,mPEAKS,NBINS_PEAKS \
                                                        ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                        ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                                                        ,mSEMANTIC,SEMANTIC_LABELS).clone();*/
                
                
                
                //evaluate
               /* float response = SVM.predict(desID);//,true);
                printf("RESPONSE SVM id: %d  %f\n",id,response);
                
                //paint image with SVM response
                if (response >= LABEL_TEXT)
                    imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();//*/
                
                if ((char)k != 'c')
                    k = waitKey(0);

            
            }
            
            desID.release();
            
            
            
        }//for id SPCTE//*/
        
        printf("==================================================\n");
        printf("**** \tTIME %s %f seconds\n",parameters["svmOptions"].as<std::string>().c_str(),(float) (((double)(clock() - start)) / CLOCKS_PER_SEC));
        printf("==================================================\n");
        printf("**** Superpixels: %f seconds\n ",SPCTE->timeSuperpixels);
        if (mLAB == 1)      printf("**** LAB: %f seconds\n ",SPCTE->timeLAB);
        if (mRGB == 1)      printf("**** RGB: %f seconds\n ",SPCTE->timeRGB);
        
        if (mEDGES == 1)    printf("**** EDGES: %f seconds\n ",SPCTE->timeEDGES);
        if (mCAFFE == 1)    printf("**** CAFFE: %f seconds\n ",SPCTE->timeCAFFE);
        if (mSEMANTIC == 1) printf("**** SEMANTIC: %f seconds\n ",SPCTE->timeSEMANTIC);//*/

        
        if (DEBUG == 1){
            imshow(nameSVM,imgSP);waitKey(0);}
        else{
            
            size_t found1 = nameImage.find_last_of("/");
            string name = "train/test/"+ nameImage.substr(found1+1) + "_" +parameters["svmOptions"].as<std::string>()+ "_" + to_string(nSAMPLES)+ ".png";
            imwrite(name,imgSP);
        }
        
        imgSP.release();
        
        delete SPCTE;

    }//svmTest

    
   // if (mCAFFE == 1) delete _caffe;
    
    
    return 0;
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TRAIN SVM

Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges, string nameSegmen)
{
    //show info
    
    /*if (DEBUG)
    {*/
        string info= "";
        if (mLAB != 0)
            info += "LAB ";
       
        if (mRGB != 0)
            info += "RGB ";
        
        if (mPEAKS != 0)
            info += "PEAKS ";
        
        if (mEDGES != 0)
            info += "EDGES  ";
        
        if (mCAFFE != 0)
            info += "CAFFE ";
        
        if (mSEMANTIC != 0)
             info += "SEMANTIC";
        printf("descriptors: %s id %d/%d \n",info.c_str(),id,SPCTE->maxID);
    //}
    
    //EDGES
    Mat imgEdges = Mat();
    if (mEDGES == 1)
    {
        // string nameEdges = dir_edges + "/" + i->path().filename().string();
        // nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
        imgEdges = imread(nameEdges,CV_LOAD_IMAGE_COLOR);
        if (imgEdges.data == NULL)
        {
            printf("ERROR: No edges image found\n");
        }
    }
    
    //SEMANTIC
    Mat imgSegmen = Mat();
    if (mSEMANTIC == 1)
    {
        imgSegmen = imread(nameSegmen,CV_LOAD_IMAGE_COLOR);
        if (imgSegmen.data == NULL)
        {
            printf("ERROR: No SEMANTIC SEGMENTATION image found\n");
        }
    }


    Mat des = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                          ,mLAB,NBINS_L,NBINS_AB \
                                          ,mRGB,NBINS_RGB\
                                          ,mPEAKS,NBINS_PEAKS \
                                          ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                          ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                                          ,mSEMANTIC,SEMANTIC_LABELS,imgSegmen).clone();
    //caffe
    
    if (mCAFFE == 1)
    {
        clock_t start = clock();
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1).clone();
        
        Mat desCaf= _caffe->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        /* double min, max;
         minMaxLoc(desCaf, &min, &max);
         Scalar     mean, stddev;
         meanStdDev ( desCaf, mean, stddev );
         
         printf("Caffe des %d values[%f,%f] mean %f stdDev %f \n",i, min, max,mean.val[0],stddev.val[0]);*/
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);//_arraySP[i].descriptorsCAFFE(imageSP,CAFFE_LAYER,NUMCAFFE), des,des);
        else
            des=desCaf.clone();//_arraySP[i].descriptorsCAFFE(imageSP,CAFFE_LAYER,NUMCAFFE).clone();
        
        //printf("Descriptors Caffe %d\n",i);
        SPCTE->timeCAFFE += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
        
        //imageSP.release();
        //desCaf.release();*/
        
    }
    
    return des;

}

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges,string dir_segmen, int nSAMPLES, string nameSVM, string out)
{
    
    //SVM
    int numDES = 0;
    
    if (mLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
    if (mRGB == 1)      numDES += (3*NBINS_RGB);
    if (mPEAKS == 1)    numDES += NBINS_PEAKS;
    if (mEDGES == 1)    numDES += NBINS_EDGES;
    if (mCAFFE == 1)    numDES += NUMCAFFE;
    if (mSEMANTIC == 1) numDES += SEMANTIC_LABELS;
    
    
    Mat trainingData = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    Mat labels = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
    
    int pos=0;
    int neg=0;
    
    int numI=0;
    
    for (auto i = directory_iterator(dir_path); i != directory_iterator() && (pos+neg < nSAMPLES*2); i++)
    {
        if (!is_directory(i->path()))
        {
            if ((pos+neg >= nSAMPLES*2))
                break;
            
            //string IMAGE
            string nameImage = dir_path + "/" + i->path().filename().string();
            string extension = i->path().filename().extension().string();
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                printf("\n\n==================================================\n");
                printf("**** Image (%d): %s (%d / 1000)\n",(pos + neg),i->path().filename().string().c_str(),++numI);
                printf("==================================================\n");
                
                //string GT
                string imageGT = dir_pathGT + "/gt_" + i->path().filename().string();
                size_t lastindex = imageGT.find_last_of(".");
                imageGT = imageGT.substr(0, lastindex) + ".png";
                
                //string EDGES
                string nameEdges = "";
                if (mEDGES == 1)
                {
                    nameEdges = dir_edges + "/" + i->path().filename().string();
                    nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                }
                
                //string SEMANTIC
                string nameSegmen = "";
                if (mSEMANTIC == 1)
                {
                    nameSegmen = dir_segmen + "/" + i->path().filename().string();
                    nameSegmen = nameSegmen.substr(0, nameSegmen.find_last_of(".")) + ".png";
                }
                
                //SUPERPIXELS
                SuperPixels *SPCTE;
                
                SPCTE=svmSuperpixelsTEXT(nameImage,2,imageGT,nameSegmen);
                

                //for each id in the image
                for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                {
                    
                    //Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                   
                    //ADD desID to SVM and labels
                    int n = neg + pos;
                    
                    if ((SPCTE->_arraySP[id].accLabel(1) == 1.0) && (pos < nSAMPLES))
                    { //text
                        Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                        labels.at<float>(n,0) = (float) LABEL_TEXT;
                        desID.row(0).copyTo(trainingData.row(n));
                        pos = pos + 1;
                        desID.release();
                    }
                    else
                    {
                        if ((SPCTE->_arraySP[id].accLabel(0) == 1.0) && (neg < nSAMPLES))
                        {
                            Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                            labels.at<float>(n,0) = (float) LABEL_NOTEXT;
                            //add des in trainingData
                            desID.row(0).copyTo(trainingData.row(n));
                            neg = neg + 1;
                            desID.release();
                        }
                    }
                    //desID.release();
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
    
    SVM.clear();
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

