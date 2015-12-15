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

#define TEST 0

//0 tsukuba
//1 venus
//2 teddy
//3 otro
// 5 rocas

Mat cmpImages( String nameGT,String name1,String label)
{
    //cargar las imágenes en GRAY
    
    Mat gt = imread(nameGT,CV_LOAD_IMAGE_GRAYSCALE);
    Mat im1 = imread(name1,CV_LOAD_IMAGE_GRAYSCALE);
    
    
   // imshow("gt",gt);
     imshow("im",im1);
    
    Mat diff = abs(gt - im1);
   
    Scalar cero(0,0,0);
    diff.setTo(cero,(gt == 0));
    
    int total= sum(diff).val[0];
    
    
    Scalar mean1, std1;
    Mat mask = (gt != 0);
    meanStdDev(diff,mean1,std1);//,mask);
    
    
    printf("--------------------------------------------\n");
    printf("* %s\n",name1.c_str());
    printf("--------------------------------------------\n");
    printf("          |   total    |   mean  |     std      |\n");
    printf("--------------------------------------------\n");
    printf(" %s | %0.2f | %5.4f | %0.4f  |\n",
           label.c_str(),(float)total/float(gt.cols*gt.rows*255),float(mean1.val[0]),float(std1.val[0]));
    printf("--------------------------------------------\n");
    // imwrite(name1 + "_diff.png",diff);
    
    return diff;
    
}


int main(int argc, const char * argv[]) {
    
    //SuperPixels *SP;
    SuperPixels *SPCTE;
    
    string nameImage = argv[1];
    
    SPCTE = new SuperPixels(nameImage);
    
    //check neigbour
   // Mat image
    for (int id=0; id < SPCTE->maxID+1; id++)
    {
        imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
        waitKey(25);
    }
    
    return 0;
    
    
    
    String nameInput;// = "tsu-ICM.png";//"col3_disp.png";//"tsu-ICM.png";//"depth_img-1.png";// "rocas_userInput.png";// "/Users/acambra/Desktop/userInput_p.png";
    String lineal, cte;

    
    if (TEST == 0)//tsukuba
    {
        nameImage = "/Users/acambra/Dropbox/testDenseLabelingUser/tsukuba/col3.png";
        nameInput = "/Users/acambra/Dropbox/testDenseLabelingUser/tsukuba/tsukuba_userInput_p.png";
        //depth/col3_disp.png";
        //lineal = "/Users/acambra/TESIS/images_test/resul_tsukuba/col3_lineal copia.png";
        cte = "/Users/acambra/Dropbox/testDenseLabelingUser/tsukuba/params/userInput_20_05_09_60.png";
        
    }else if (TEST == 1) // venus
    {
        nameImage = "/Users/acambra/Dropbox/testDenseLabelingUser/venus/imL.png";
        nameInput = "/Users/acambra/Dropbox/testDenseLabelingUser/venus/user/venus_userInput_p.png";//disp/imL_disp.png";
       // lineal = "venus/imL_lineal.png";
        cte = "/Users/acambra/Dropbox/testDenseLabelingUser/venus/params/userInput_20_05_09_23.png";
        
    }else if (TEST == 2) // teddy
    {
        nameImage = "/Users/acambra/Dropbox/testDenseLabelingUser/teddy/im2.png";
        nameInput = "/Users/acambra/Dropbox/testDenseLabelingUser/teddy/teddy_userInput_p2.png";//im2_disp.png";
       // lineal = "/Users/acambra/TESIS/images_test/resul_teddy/im2_lineal copia.png";
        cte = "/Users/acambra/Dropbox/testDenseLabelingUser/teddy/params/userInput_20_05_09_100.png";
        //disp/out_20.png";
        
    }
    else if (TEST == 5)//rocas
    {
        nameImage = "/Users/acambra/Dropbox/Tesis Compartida/DenseLabelingLatex/images/rocas/rocas.jpg";
        nameInput = "/Users/acambra/Dropbox/Tesis Compartida/DenseLabelingLatex/images/rocas/userInput/userInput.png";
        cte = "/Users/acambra/Dropbox/Tesis Compartida/DenseLabelingLatex/images/rocas/param/userInput_2_04_099.png";
        
    }
    else
    {
        nameImage = "/Users/acambra/TESIS/images_test/resul_venus/imL.png";//;// "RGB_img-1.png";
        nameInput = "/Users/acambra/Google Drive/venus_userInput_p2.png";//"/Users/acambra/TESIS/images_test/resul_venus/venus_userInput_p2.png";
        //;WTA_Birchfield.png";//tsukuba_userInput_p.png";//depth_img-1.png";
        lineal = "/Users/acambra/TESIS/images_test/resul_venus/lineal2.png";//"img-1_lineal.png";
        cte = "/Users/acambra/TESIS/images_test/resul_venus/cte_user2.png";//"img-1_cte.png";
        
    }

    //cargar imagen original
  /*  SP = new SuperPixels(nameImage);
    
    SP->loadSuperPixels("");
    
    SP->loadDepthBuildSystemCoef(nameInput);
    
    SP->addEquationsBinariesBoundariesCoef();
    SP->addEquationsUnariesMedianCoef();//addEquationsUnariesCoef();//
    SP->solveCoef();
    
   /* Mat input;
    input=imread(nameInput);
    imshow("Input",input);
    imwrite(lineal,SP->getImage());//*/
    //waitKey(0);
    
     SPCTE = new SuperPixels(nameImage);
     //SPCTE->loadSuperPixels("");
   // imshow("Superpixel",SPCTE->getImage());
   // waitKey(0);

    /* SPCTE->loadDepthBuildSystem(nameInput);
     SPCTE->addEquationsBinariesBoundaries(false);
    
     SPCTE->addEquationsUnaries();
   
     SPCTE->solve();*/
    //imwrite(cte,SPCTE->getImage());//*/

    imshow("superpixels",SPCTE->getImage());
    waitKey(0);
    return 0;
    
    
    
    
    /*clock_t start = clock();
    
    Mat image;
    cvtColor(SPCTE->_lab,image,CV_Lab2BGR);
    Mat imageDepth = SPCTE->_pixelDepth*255.0;
    
  /*  Mat final;
    
    
    int   nbins          = 8;
    float aperture       = 7.0;
    float focal_distance = 220;
    float focal_length   = 220+20;
    bool  linear         = true;
    Mat b;
    
    //clock_t
    start = clock();
    
    b=SPCTE->blurImageDepth(image, imageDepth, nbins,focal_distance,focal_length,aperture, linear);
    
    
    //imwrite("blur.png", b);
    
    printf("**** TIEMPO: Blur: %f seconds\n ",(float) (((double)(clock() - start)) / CLOCKS_PER_SEC) );*/
    //waitKey(0);
    
   String nameGT="/Users/acambra/TESIS/images_test/tsukuba/gt/gt.png";
    String mrf = "/Users/acambra/TESIS/images_test/tsukuba/gt/gt.png";//Expansion_p2.png";
    
    Mat diff= cmpImages(nameGT,mrf, "label");
    //imshow("diff",diff);
    //waitKey(0);
    return;//*/
    
    return 0;
}