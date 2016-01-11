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
    
    //SuperPixels *SP;
    SuperPixels *SPCTE;
    
    string nameImage = argv[1];
    
    SPCTE = new SuperPixels(nameImage);
    
    //init superpixels
    SPCTE->calculateBoundariesSuperpixels();

    SPCTE->initializeSuperpixels();
    
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