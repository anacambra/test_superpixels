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


int main(int argc, const char * argv[]) {
    
    SuperPixels *SPCTE;
    
    string nameImage = argv[1];
    
    SPCTE = new SuperPixels(nameImage);
   
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    
    //init superpixels
    SPCTE->initializeSuperpixels();
   /* imshow("superpixels",SPCTE->getImageSuperpixels());
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
    int numDES = 256;
    int nSAMPLES = 5;
    Mat trainingData = Mat::zeros(SPCTE->maxID+1,numDES,CV_32FC1);
    Mat labels = Mat::zeros(SPCTE->maxID+1, 1, CV_32FC1);
    
    
    int pos=0;
    int neg=0;
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        
        //imshow("crop image",SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,10));
        // imshow("crop labels",SPCTE->cropSuperpixel(out,id,10));
        Mat des = SPCTE->calculateDescriptors(id, out, 0,0,0,0,0,1,numDES).clone();
        
        //add des in trainingData
        //des.row(0).copyTo(trainingData.row(id));
        
        if (SPCTE->_arraySP[id].accLabel(1 && pos < nSAMPLES) >= 0.5)
        
        { //text
            labels.at<float>(id,1) = 10.0;
            pos= pos +1;
             printf("SVM labels  %f\n",labels.at<float>(id,1));
            //add des in trainingData
            des.row(0).copyTo(trainingData.row(id));
        }else
         if (neg < nSAMPLES){
            labels.at<float>(id,1) = -1.0;
             neg=neg+1;
              printf("SVM labels  %f\n",labels.at<float>(id,1));
             //add des in trainingData
             des.row(0).copyTo(trainingData.row(id));
        }
        
    }
    //train SVM
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingData, labels, Mat(), Mat(), params);
    
    //test
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        
        imshow("crop image",SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
        imshow("crop labels",SPCTE->cropSuperpixel(out,id,3));
        Mat desID = SPCTE->calculateDescriptors(id,out,0,0,0,0,0,1,numDES).clone();
        
        //evaluate
        float response = SVM.predict(desID);
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