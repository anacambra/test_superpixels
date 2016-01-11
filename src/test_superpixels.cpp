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
   
    //boundaries betweenSP
    SPCTE->calculateBoundariesSuperpixels();
    
    //init superpixels
    SPCTE->initializeSuperpixels();
    
    //labels
   // imshow("superpixels",SPCTE->getImageSuperpixels());
   // imshow("in labels",imread(argv[2]));
           
    
    //TEXT LABELS
    SPCTE->setNUMLABELS(60);
    //init labeling
    Mat out = SPCTE->initializeMeanLabeling(argv[2]).clone();
    imshow("labels", out);
    
    //check superpixel
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        imshow("crop in labels", SPCTE->cropSuperpixel(imread(argv[2]),id,3));
        imshow("crop labels",SPCTE->cropSuperpixel(out,id,3));
        imshow("crop image",SPCTE->cropSuperpixel(SPCTE->getImage(),id,3));
        
        
        
        waitKey(0);
    }//*/
    
    
    
    
    
    
    
    
    SPCTE->calculateDescriptors();
    
    imshow("superpixels",SPCTE->getImageSuperpixels());
    
    
    //check neigbour
   // Mat image
    /*for (int id=0; id < SPCTE->maxID+1; id++)
    {
        imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
        waitKey(25);
    }*/
    
    
    //init labeling
   /* Mat out = SPCTE->initializeMeanLabeling(argv[2]).clone();
    
    imshow("labels", out);*/
    waitKey(0);
    
    
    
    return 0;
    
}