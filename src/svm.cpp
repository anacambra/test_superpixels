//
//  svm.cpp
//  test_sup
//
//  Created by Ana Cambra on 9/1/16.
//
//

#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

int svm()
{
    
    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    
    //
    /*Mat sampleMat = (Mat_<float>(1,2) << j,i);
    float response = SVM.predict(sampleMat);
    
    if (response == 1)
    image.at<Vec3b>(i,j)  = green;
    else if (response == -1)
    image.at<Vec3b>(i,j)  = blue;*/

}