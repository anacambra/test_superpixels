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

void descriptorSVMText(string dir_path)
{
    
    //SVM
    int numDES = 128;
    int nSAMPLES = 150;
    
    Mat trainingData = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    Mat labels = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
    
    int pos=0;
    int neg=0;
    
    
    for (auto i = directory_iterator(dir_path); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path()))
        {
            if ((pos+neg < nSAMPLES*2))
                break;
            
            string nameImage = dir_path + "/" + i->path().filename().string();
            string extension = i->path().filename().extension().string();
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                SuperPixels *SPCTE;
                SPCTE = new SuperPixels(nameImage);
                //boundaries between SP
                SPCTE->calculateBoundariesSuperpixels();
                //init superpixels
                SPCTE->initializeSuperpixels();
                //TEXT LABELS
                SPCTE->setNUMLABELS(2);//atoi(2));
                
                for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                {
                    Mat des = SPCTE->calculateDescriptors(id, SPCTE->getImage(), 0,0,0,0,0,1,numDES).clone();
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
    string filename = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/svm_train.yaml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "trainingData" << trainingData;
    fs << "labels" << labels;
    fs.release();
    cout << trainingData << endl;
    
}

//for image in dir
//superpixels
//for id in superpixels
//descriptor in trainingData
//labels

//

int main(int argc, const char * argv[]) {
    
    
    descriptorSVMText("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_images");
    
    SuperPixels *SPCTE;
    
    string nameImage = argv[1];
    
    SPCTE = new SuperPixels(nameImage);
   
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    
    //init superpixels
    SPCTE->initializeSuperpixels();
    imshow("superpixels",SPCTE->getImageSuperpixels());
    waitKey(0);//*/
    
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
    Mat out = SPCTE->initializeLabeling(argv[2], MODE_LABEL_NOTZERO).clone();
    //imshow("labels", out);
    
    //check superpixel
    Mat im=out;
    
    //SVM
    int numDES = 4;
    int nSAMPLES = 6;
    
    Mat trainingData = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    Mat labels = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
    
   
    
    int pos=0;
    int neg=0;
    
    for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
    {
        //imshow("crop image",SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,10));
        // imshow("crop labels",SPCTE->cropSuperpixel(out,id,10));
        
        //Mat des = SPCTE->calculateDescriptors(id, out, 0,0,0,0,0,1,numDES).clone();
        Mat des = SPCTE->calculateDescriptors(id, SPCTE->getImage(), 0,0,0,0,0,1,numDES).clone();
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
                
               // printf("-%d SVM labels  %f\n",n,labels.at<float>(n,0));
                //add des in trainingData
                des.row(0).copyTo(trainingData.row(n));
                neg = neg + 1;
                //
            }
            
        }
    }
    
    
    
    
    //SAVE TRAIN DESCRIPTORS
    string filename = argv[4];
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "trainingData" << trainingData;
    fs << "labels" << labels;
    fs.release();
    cout << trainingData << endl;
    
    //READ TRAIN DESCRIPTORS
   // string filename = argv[4];
    FileStorage fsREAD(filename, FileStorage::READ);
    
    Mat labels2 = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
     Mat trainingData2 = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    fsREAD["labels"] >> labels2;
    fsREAD["trainingData"] >> trainingData2;
    fsREAD.release();
    
    
    cout << trainingData2;
    return 0;

    //train SVM
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
    
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingData, labels, Mat(), Mat(), params);
    
    //test
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        
        imshow("crop image",SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
        imshow("crop labels",SPCTE->cropSuperpixel(out,id,3));
        //Mat desID = SPCTE->calculateDescriptors(id,out,0,0,0,0,0,1,numDES).clone();
        Mat desID = SPCTE->calculateDescriptors(id,SPCTE->getImage(),0,0,0,0,0,1,numDES).clone();
        
        //evaluate
        float response = SVM.predict(desID,true);
        printf("RESPONSE SVM  %f\n",response);
        
        char k = waitKey(0);
       
        if(k == 27) break; //ESC
        else  if(k == 2) //<-
            if (id > 2) id= id - 2;
       // else  if(k == 3) //->
        //*/
        
    }//*/
    
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
    
    waitKey(0);
    
    
    
    return 0;
    
}