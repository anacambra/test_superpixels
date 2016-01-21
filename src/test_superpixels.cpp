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

using namespace boost::filesystem;

#define LABEL_TEXT 1
#define LABEL_NOTEXT -1

#define DEBUG_SP 0

int mLAB   = 1; int NBINS_L = 50; int NBINS_AB=128;
int mRGB   = 1; int NBINS_RGB   = 256;
int mPEAKS = 1; int NBINS_PEAKS = 128;

int mLINES = 1; int NBINS_LINES = 128;

string nameSVM = "svm_LAB_RGB_PEAKS.xml";

void descriptorSVMText(string dir_path,string dir_pathGT, string out)
{
    
    //SVM
    int numDES = 0;
    if (mLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
    if (mRGB == 1)      numDES += (3*NBINS_RGB);
    if (mPEAKS == 1)    numDES += NBINS_PEAKS;
    
    int nSAMPLES = 50;
    
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
                
                SPCTE->initializeLabeling(imageGT, MODE_LABEL_MEDIAN);//MODE_LABEL_NOTZERO);
                
                for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                {
                    Mat des = SPCTE->calculateDescriptors(id, SPCTE->getImage(), mLAB,NBINS_L,NBINS_AB,mRGB,NBINS_RGB,mPEAKS,NBINS_PEAKS).clone();
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
                            //
                        }
                        
                    }
                }
            }
        }
    }//for
    
    //SAVE TRAIN DESCRIPTORS
   // string filename = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/svm_train.yaml";
    FileStorage fs(out, FileStorage::WRITE);
    fs << "trainingData" << trainingData;
    fs << "labels" << labels;
    fs.release();
    cout << trainingData << endl;
    
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

//for image in dir
//superpixels
//for id in superpixels
//descriptor in trainingData
//labels

//

int main(int argc, const char * argv[]) {
    
    
    ifstream infile(nameSVM);
    if (! infile.good()) {
        descriptorSVMText("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images",
                          "/Users/acambra/Dropbox/dataset/ICDAR/masks",
                          "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/des_"+nameSVM+".yaml");
    }
    
    //evaluate
    CvSVM SVM;
    SVM.load(nameSVM.c_str());
    
    //SUPERPIXELS
    
    SuperPixels *SPCTE;
    
    string nameImage = argv[1];
    
    SPCTE = new SuperPixels(nameImage);
   
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    
    //init superpixels
    SPCTE->initializeSuperpixels();
    Mat imgSP = SPCTE->getImageSuperpixels().clone();
    imshow("superpixels",SPCTE->getImageSuperpixels());
    //waitKey(0);//*/
    
    //check neigbour
    // Mat image
    /*for (int id=0; id < SPCTE->maxID+1; id++)
     {
     imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
     waitKey(25);
     }*/
    
    //TEXT LABELS
    SPCTE->setNUMLABELS(atoi(argv[3]));
    //init labeling
    Mat gt = SPCTE->initializeLabeling(argv[2], MODE_LABEL_NOTZERO).clone();
    imshow("GT", gt);
    char k=waitKey(1);
    
    //test
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        //Mat desID = SPCTE->calculateDescriptors(id,out,0,0,0,0,0,1,numDES).clone();
        Mat desID = SPCTE->calculateDescriptors(id,SPCTE->getImage(),mLAB,NBINS_L,NBINS_AB,mRGB,NBINS_RGB,mPEAKS,NBINS_PEAKS).clone();
        
        
        
        //evaluate
        float response = SVM.predict(desID);//,true);
        printf("RESPONSE SVM  %f\n",response);
       // k=waitKey(0);
        if (DEBUG_SP == 1) {
            imshow("Superpixel i",SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
            imshow("GT label",SPCTE->cropSuperpixel(gt,id,3));
            
            //test LINES
            Mat lines= SPCTE->_arraySP[id].descriptorsLINES(SPCTE->getImage(),NBINS_LINES).clone();
            
            imshow("lines i",SPCTE->cropSuperpixel(lines,id,3));
            
            
            if(k == 27){ break;} //ESC
            else{
                if ((k == 2) && (id > 2)){//<-
                    id= id - 2;
                    k = waitKey(0);
                }
                else  if (k == 3) k = waitKey(0);
                else  if(k == 80 || (k == 112)) k = waitKey(0); // p
                else
                    k = waitKey(1);
            }
            //*/
        }
        //paint image with SVM response
        if (response >= LABEL_TEXT)
        imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();
        
        
    }//*/
    
    imshow("SVM response",imgSP);waitKey(0);
    
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