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
    
    imshow("superpixels",SPCTE->getImageSuperpixels());
    
    //inti labeling
    SPCTE->initializeMeanLabeling(argv[2]);
    
    //check neigbour
   // Mat image
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
        waitKey(25);
    }
    
    return 0;
    
}