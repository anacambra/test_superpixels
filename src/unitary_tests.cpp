//
//  unitary_tests.c
//  tests
//
//  Created by Ana B. Cambra on 06/05/16.
//  Copyright (c) 2016 Ana B. Cambra. All rights reserved.
//

#include <stdio.h>

#include "superpixels.cpp"
#include "utilsCaffe.cpp"
//#include "Descriptors.cpp"

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include <stdio.h>

#include <random>


class UnitaryTests
{
    
public:
    ////////////////////// TEST DESCRIPTORS
    static void descriptorsLAB(string type,string nameImage)
    {
        SuperPixels *SPCTE;
        char k=-1;
        
        //init superpixels
        SPCTE = new SuperPixels(nameImage);
        SPCTE->calculateBoundariesSuperpixels();
        SPCTE->initializeSuperpixels();
        
        Mat img = imread(nameImage,CV_LOAD_IMAGE_COLOR);
        imshow("superpixels",SPCTE->getImageSuperpixels());
        
        if (type.find("LAB") != std::string::npos)
        {

            for (int id=0; id < SPCTE->maxID+1; id++)
            {
                Mat mask = SPCTE->getMask(id);
                //KEYBOARD
                if (k == 2){//<-
                    if (id >= 2) id = id - 2;
                    if (id == 1) id = 0;
                }
                if (k == 27){//ESC
                    break;
                }
                
                //CONLAB
                /*Mat imageSP = SPCTE->cropSuperpixel(img,id,1,2).clone();
                imshow("CONLAB",imageSP);
                
                // id with mask superpixel
                Descriptors::descriptorsLAB(img,SPCTE->getMaskNeigboursBB(id),50,100); //*/
                
                //LAB
                clock_t start= clock();
                Mat imageSP = SPCTE->cropSuperpixel(img,id,1,0).clone();
                Descriptors::descriptorsLAB(imageSP,Mat(),50,100);
                cout<< "CROP: "<< (float) (((double)(clock() - start)) / CLOCKS_PER_SEC) <<endl;
                imshow("LAB",imageSP);
                 
                 // id with mask superpixel
                start= clock();
                Descriptors::descriptorsLAB(img,SPCTE->getMaskBB(id),50,100); //*/
                cout<< "WITH MASKS " << (float) (((double)(clock() - start)) / CLOCKS_PER_SEC) <<endl;
                
                if ((char)k != 'c')
                    k = waitKey(0);
                
            }
        }
        delete SPCTE;
    }
    
    //////////////////////
    static void descriptorsEDGES(string type,string nameImage)
    {
        SuperPixels *SPCTE;
        char k=-1;
        
        //init superpixels
        SPCTE = new SuperPixels(nameImage);
        SPCTE->calculateBoundariesSuperpixels();
        SPCTE->initializeSuperpixels();
        
        Mat img = imread(nameImage,CV_LOAD_IMAGE_COLOR);
        imshow("superpixels",SPCTE->getImageSuperpixels());
        
        if (type.find("EDGES") != std::string::npos)
        {
            string nedges =nameImage.replace(nameImage.find("/images/"), sizeof("/edges/"), "/edges/");
            
            nedges =nedges.replace(nedges.find(".jpg"), sizeof(".png"), ".png");
            
            Mat edges = imread(nedges,CV_LOAD_IMAGE_COLOR);

            //show edges
            for (int id=0; id < SPCTE->maxID+1; id++)
            {
                Mat mask = SPCTE->getMask(id);
                //KEYBOARD
                if (k == 2){//<-
                    if (id >= 2) id = id - 2;
                    if (id == 1) id = 0;
                }
                if (k == 27){//ESC
                    break;
                }
                
                int modeEDGES=1;
                
                //EDGES
                clock_t start= clock();
                Descriptors::descriptorsEDGES(edges,SPCTE->getMaskBB(id),16,modeEDGES); //imagen
                cout<< "WITH MASK CROP: "<< (float) (((double)(clock() - start)) / CLOCKS_PER_SEC) <<endl;
                
                //edges id with mask superpixel
                
                //edges id cropped superpixel BB
                start= clock();
                Mat imageSP = SPCTE->cropSuperpixel(edges,id,1,0).clone();
                Descriptors::descriptorsEDGES(imageSP,Mat(),16,modeEDGES); //imageSP //*/
                cout<< "CROP: "<< (float) (((double)(clock() - start)) / CLOCKS_PER_SEC) <<endl;
                imshow("edges",imageSP);
                //CONEDGES
               /* Mat imageSP = SPCTE->cropSuperpixel(edges,id,1,2).clone();
                imshow("edges",imageSP);
                //edges id with mask superpixel
                Descriptors::descriptorsEDGES(edges,SPCTE->getMaskNeigboursBB(id),8,modeEDGES); //imagen
                
                //edges id cropped superpixel BB
                Descriptors::descriptorsEDGES(imageSP,Mat(),8,modeEDGES); //imageSP//*/

              
                if ((char)k != 'c')
                    k = waitKey(0);
                
            }
        }
        delete SPCTE;
    }
    
    //SHOW SUPERPIXELS
    //////////////////////
    static void showSuperpixels(string nameImage)
    {
        SuperPixels *SPCTE;
        char k=-1;
        
        SPCTE = new SuperPixels(nameImage);
        
        //boundaries between SP
        SPCTE->calculateBoundariesSuperpixels();
        
        imshow("superpixels",SPCTE->getImageSuperpixels());
        k = waitKey(0);
        
        SPCTE->initializeSuperpixels();
        
        for (int id=0; id < SPCTE->maxID+1; id++)
        {
            //KEYBOARD
            if (k == 2){//<-
                if (id >= 2) id = id - 2;
                if (id == 1) id = 0;
            }
            if (k == 27){//ESC
                break;
            }
            imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
            
            imshow("BB",SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,0).clone());
            imshow("BB with mask",SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,1).clone());
            imshow("BB Neig with mask",SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,2).clone());
            
            if ((char)k != 'c')
                k = waitKey(0);
            
        }
        delete SPCTE;
    }
    
    
    //////////////////////
    static void cmpResults(string dir,string dir1, string dir2, string dirGT,string dirCMP)
    {
        Mat img,gt, img1, img2,out;
        
        
        printf("     IMAGE                    i           total_px          svm_px     svm_%%          segnet_px    segnet_%%\n");
        printf("--------------------------------------------------------------------------------------------------------------------");
        for (auto i = boost::filesystem::directory_iterator(dir); i != boost::filesystem::directory_iterator(); i++)
        {
            if (!is_directory(i->path()))
            {
                //string IMAGE
                string nameImage = i->path().filename().string();
                string extension = i->path().filename().extension().string();
                
                
                if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
                {
                    string image = dir + "/" + nameImage ;
                    size_t lastindex = nameImage.find_last_of(".");

                    string imageGT = dirGT + "/gt_" + nameImage.substr(0, lastindex) + ".png";
                    string gtTXT = dirGT + "/gt_" + nameImage.substr(0, lastindex) + ".txt";
                    string image1 = dir1 + + "/" + nameImage + "__LAB7_EDGES_CAFFE_SEMANTIC_CONLAB7_CONEDGES_CONCAFFE2_CONTEXT2_570_RBF.png";
                    string image2 = dir2 + + "/" + nameImage.substr(0, lastindex) + "_SegNet.png";
                    
                    /*cout << image << endl;
                    cout << imageGT << endl;
                    cout << image1 << endl;
                    cout << image2 << endl;//*/
                    
                    img = imread(image,CV_LOAD_IMAGE_COLOR);
                    img1 = imread(image1,CV_LOAD_IMAGE_COLOR);
                    img2 = imread(image2,CV_LOAD_IMAGE_COLOR);
                    gt = imread(imageGT,CV_LOAD_IMAGE_COLOR);
                    
                    if (img.data == NULL || img1.data == NULL || img2.data == NULL || gt.data == NULL )
                        return ;
                    
                    if (gt.rows != img.rows || gt.cols != img.cols)
                        resize(gt, gt, img.size());

                    if (img2.rows != img.rows || img2.cols != img.cols)
                        resize(img2, img2, img.size());
                    
                    
                    // img1 => img1_mask 8UC1
                    //get mask img1
                    vector<Mat> bgr;
                    split(img1,bgr);
                    
                    Mat img1_mask;
                    Mat img1_b=(bgr[0] == 0);
                    Mat img1_g=(bgr[1] == 0);
                    
                    bitwise_and(img1_b, img1_g, img1_mask);
                    Mat img1_r = (bgr[2] == 255);
                    bitwise_and(img1_mask, img1_r, img1_mask);
                    
                    img1_b.release();
                    img1_g.release();
                    img1_r.release();
                    
                    // img2 => img2_mask 8UC1
                    Mat img2_mask;
                    cvtColor(img2, img2_mask, CV_BGR2GRAY);

                    //hipotesis GT
                    FILE *fp;
                    
                    if( (fp = fopen(gtTXT.c_str(), "r+")) == NULL)
                    {
                        exit(1);
                    }
                    
                    int pts[8]={0,0,0,0,0,0,0,0};
                    
                    int num=0;
                    
                    while (fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%*[^\n]",&pts[0], &pts[1], &pts[2], &pts[3],
                                  &pts[4], &pts[5], &pts[6], &pts[7]) != EOF)//!feof(fp) )
                    {

                        Mat hipGT = Mat::zeros(img.rows,img.cols,CV_8UC1);
                        
                        Point points[1][4];
                        points[0][0] = Point(pts[0],pts[1]);
                        points[0][1] = Point(pts[2],pts[3]);
                        points[0][2] = Point(pts[4],pts[5]);
                        points[0][3] = Point(pts[6],pts[7]);
                        
                        const Point* ppt[1] = { points[0] };
                        int npt[] = { 4 };
                        fillPoly( hipGT,
                                 ppt,
                                 npt,
                                 1,
                                 Scalar( 255, 255, 255 ),
                                 CV_AA );
                        
                        
                        
                        
                        //AND
                        Mat hipGT1,hipGT2;
                        Mat hipOUT = img.clone();
                        
                        bitwise_and(hipGT,img1_mask , hipGT1);
                        bitwise_and(hipGT,img2_mask , hipGT2);
                        //imshow("hipGT1",hipGT1);
                        //imshow("hipGT2",hipGT2);
                        
                        int num1,num2,numTotal;
                        
                        num1 = countNonZero(hipGT1);
                        num2 = countNonZero(hipGT2);
                        numTotal = countNonZero(hipGT);
                        
                        /*printf("%.12d (%f) SegNet: %d (%f) GT: %d\n",
                               num1,,
                               num2, ((float)num2/ (float)numTotal),
                               numTotal);//*/
                        
                        
                        polylines(hipOUT, ppt,npt, 1,
                                  true, 			// draw closed contour (i.e. joint end to start)
                                  Scalar(0,255,0),// colour RGB ordering (here = green)
                                  5, 		        // line thickness
                                  CV_AA, 0);
                        
                        Rect bb2 = SuperPixels::boundingbox(hipGT2);
                        rectangle(hipOUT,bb2, Scalar(0,0,255),2,CV_AA); //SegNet RED

                        Rect bb1 = SuperPixels::boundingbox(hipGT1);
                        rectangle(hipOUT,bb1, Scalar(255,0,0),2,CV_AA); //SVM BLUE
                        
                        //imshow("img",imgOUT);
                        
                        cvtColor(hipGT,hipGT,CV_GRAY2BGR);
                        cvtColor(hipGT1,hipGT1,CV_GRAY2BGR);
                        cvtColor(hipGT2,hipGT2,CV_GRAY2BGR);
                        
                        hconcat(hipOUT,hipGT1,hipOUT);
                        hconcat(hipOUT,hipGT2,hipOUT);
                        hconcat(hipOUT,hipGT,hipOUT);
                        
                        resize(hipOUT,hipOUT, Size(480*4,360));
                        //imshow("hipOUT",hipOUT);
                        int space = 15;
                        printf("\n%-15s %*d %*d %*d %10.2f %% %*d %10.2f %%",
                                                    nameImage.c_str()
                                                    ,space,num
                                                    ,space,numTotal
                                                    ,space,num1
                                                    ,((float)num1/ (float)numTotal)*100.0
                                                    ,space,num2
                                                    ,((float)num2/ (float)numTotal)*100.0);
                        
                        
                        if (num1 > num2)
                        {
                            string nameOut = dirCMP + "/svm/" + nameImage + "_" + to_string(num)+ ".png";
                            //imwrite(nameOut,hipOUT);
                        }
                        else if (num2 > num1)
                        {
                            string nameOut = dirCMP + "/SegNet/" + nameImage + "_" + to_string(num)+ ".png";
                           // imwrite(nameOut,hipOUT);
                        }
                        
                        
                        
                        hipGT.release();
                        hipGT1.release();
                        hipGT2.release();
                        hipOUT.release();
                        
                        num++;

                    }//while fcanf txt file
                    
                    
                    //show images
                    hconcat(img,img1,out);
                    hconcat(out,img2,out);
                    hconcat(out,gt,out);
                    
                    resize(out,out, Size(480*4,360));
                    //imshow("out",out);
                    //waitKey(0);
                    
                    string nameCMP = dirCMP + "/" + nameImage + ".png";
                    //imwrite(nameCMP,out);

                }
            }
        }
        
        img.release();
        gt.release();
        img1.release();
        img2.release();
        out.release();
    
    }
    
    static void cmpLabel(string dir, string dirGT,int numLabels, int label=7)
    {
        Mat img,gt;
        
        printf(" image-SegNet      percentage          precision             recall\n");
        printf("------------------------------------------------------------------------\n");
        
        for (auto i = boost::filesystem::directory_iterator(dir); i != boost::filesystem::directory_iterator(); i++)
        {
            if (!is_directory(i->path()))
            {
                //string IMAGE
                string nameImage = i->path().filename().string();
                string extension = i->path().filename().extension().string();
                
                
                if ((nameImage.find("_rgb") == std::string::npos)
                    && (extension == ".png" || extension == ".jpg" || extension == ".jpeg"))
                {
                    string image = dir + "/" + nameImage;
                    string imageGT = dirGT + "/gt_" + nameImage.replace(nameImage.find("_SegNet.png"), sizeof("_SegNet.png")-1, ".png");

                    gt = imread(imageGT,CV_LOAD_IMAGE_GRAYSCALE);
                    img = imread(image,CV_LOAD_IMAGE_GRAYSCALE);
                    
                    img = (img / 255 )*numLabels;
                    
                    /*double min, max;
                    Point min_loc, max_loc;
                    minMaxLoc(img, &min, &max, &min_loc, &max_loc);
                    printf("%s %d %d\n",nameImage.c_str(),(int)min,(int)max);//*/
                    

                    Mat resul = (img == 7);
                   // imshow("out SIGN",resul);
                    
                    //Precision recall
                    Mat imgTP,imgFP,imgFN;
                    
                    if (gt.rows != resul.rows || gt.cols != resul.rows)
                        resize(resul, resul, gt.size());
                    
                    bitwise_and(gt,resul,imgTP);
                    
                    Mat notGT;
                    bitwise_not(gt,notGT);
                    bitwise_and(notGT,resul,imgFP);
                    
                    Mat notresul;
                    bitwise_not(resul,notresul);
                    bitwise_and(gt,notresul,imgFN);
                    
                    int tp = countNonZero(imgTP);
                    int tpfp = countNonZero(imgTP) + countNonZero(imgFP);
                    int tpfn = countNonZero(imgTP) + countNonZero(imgFN);
                    
                    float precision = (float)tp / (float)tpfp;
                    float recall = (float)tp / (float)tpfn ;
                    
                    float per= countNonZero(resul)*100.0/float(resul.cols*resul.rows);
                    
                    float precisionBB,recallBB;
                    string gtTXT =  dirGT + "/gt_" + nameImage.replace(nameImage.find(".png"), sizeof(".png")-1, ".txt");
                    cmpImageTXT(gtTXT,resul, &precisionBB, &recallBB);
                    
                    printf("%-20s%-20f%-20f%-20f%-20f%-20f\n",nameImage.c_str(),per,precision,recall,precisionBB, recallBB);
                    
                }
            }
        }
    }
    

    static void cropImageBB(string image, string out, Size outSize = Size(480,360))
    {
        Mat img = imread(image,CV_LOAD_IMAGE_COLOR);
        Mat resul = imread(out,CV_LOAD_IMAGE_UNCHANGED);
        
        utilsCaffe *caffe = new utilsCaffe("/Users/acambra/SegNet/SegNet_text/Models/Training_socarrat/train_svt1_ICDAR_iter_60000.caffemodel",\
                                           "/Users/acambra/SegNet/SegNet_text/Models/my_segnet_model_driving_webdemo_deploy.prototxt");
        
        Mat new_img,cmpIN;
        resize(img, new_img, Size(480,360),0,0,CV_INTER_AREA);
        Mat out_img = caffe->segmentation(img);
        cvtColor(out_img, out_img, CV_GRAY2BGR);
        hconcat(new_img,out_img, cmpIN);
        imshow("Input + SegNet",cmpIN);
        
        if (resul.channels() > 1)
        {
            //select red
            vector<Mat> bgr;
            split(resul,bgr);
            Mat mask=(bgr[0] == 0);//B
            Mat mask2=(bgr[1] == 0);//G
            
            bitwise_and(mask, mask2, mask);
            mask2 = (bgr[2] == 255);
            bitwise_and(mask, mask2, resul);
            //imshow("mask",resul);
            
        }
        
        RNG rng;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        
        //int numH=0;
        string nHip =out.replace(out.find("/out/"), sizeof("/out/")-1, "/cropped/");
        
        findContours( resul, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        for( int i = 0; i< contours.size(); i++ )
        {
            Mat hip = Mat::zeros(img.rows,img.cols,CV_8UC1);
           // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( hip, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 1, Point() );
            //drawContours( hip, contours, i, color, CV_FILLED, 8, hierarchy, 1, Point() );
            Point centroide;
            Moments m = moments(hip, true);
            centroide.x = m.m10/m.m00;
            centroide.y =  m.m01/m.m00;
            
            int minX,minY,w,h,maxX, maxY;
            minY = centroide.y - outSize.height/2;
            
            maxY = minY + outSize.height;
            
            minX= centroide.x - (outSize.width/2);
            maxX = minX + outSize.width;
            h = outSize.height;
            w =  outSize.width;

            
            if (minY < 0)
            {
                h = outSize.height;
                minY = 0;
                            }
            //if (minX >= img.cols)  minX = img.cols;
            
            //if (minY >= img.rows)  minY= img.rows;
            
            if (minX < 0)
            {
                minX = 0;
                
            }
            cout << minY << endl;
            cout << minX << endl;
            
            if (maxY >= img.rows)
            {
                minY = img.rows - (outSize.height);
                h =  outSize.height;
            }

            if (maxX >= img.cols)
            {
                minX = img.cols - (outSize.width);
                w =  outSize.width;
            }
            
            cout << minY << endl;
            cout << minX << endl;
            
            cvtColor(hip,hip,CV_GRAY2BGR);
            circle(hip, centroide, 3, CV_RGB(255,0,0),-1);
            rectangle(hip,
                      Rect((centroide.x - outSize.width/2), (centroide.y - outSize.height/2),outSize.width,outSize.height),
                      CV_RGB(0,255,0),1,CV_AA,0);

            Mat cropped = img( Rect(minX,minY,w,h));
            Mat out_img_resize;
            resize(out_img, out_img_resize, Size(img.cols,img.rows),0,0,CV_INTER_AREA);
            Mat out_segNet_crop = out_img_resize(Rect(minX,minY,w,h));
            //imshow("img",img);
            //imshow("hip",hip);
            //imshow("cropped",cropped);
            
            Mat out_crop = caffe->segmentation(cropped);
            cvtColor(out_crop, out_crop, CV_GRAY2BGR);
            hconcat(cropped,out_segNet_crop,cropped);
            hconcat(cropped,out_crop,cropped);
            imshow("cropped+initial cropped+ segnetCrop",cropped);
            
            waitKey(0);
            
           
            string numHfile = "_" + to_string(i) + ".png";
            string nameHipI = nHip;
            nameHipI=nameHipI.replace(nameHipI.find(".png"), sizeof(".png"), numHfile);
            
            //cout << nameHipI<< endl;
           // imwrite(nameHipI, cropped);

        }
        
        
        
    }

    static void testSegNet(string image)
    {
        utilsCaffe *caffe = new utilsCaffe("/Users/acambra/SegNet/SegNet_text/Models/Training_socarrat/train_svt1_ICDAR_iter_60000.caffemodel",\
                                           "/Users/acambra/SegNet/SegNet_text/Models/my_segnet_model_driving_webdemo_deploy.prototxt");
        
        Mat img = imread(image,CV_LOAD_IMAGE_COLOR);
        //caffe->segmentation(img);
        imshow("Mat",caffe->segmentation(img));
        waitKey(0);
    }
    
    
    
    //cmp BB precision recall
    static void cmpImageTXT(string gtTXT, Mat resul, float* precisionBB, float *recallBB)
    {
        //open file TXT
        
     //   imshow("resul",resul);
        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        
        findContours( resul, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        //int totalDet = contours.size();
        int numGT = 0;
        //int numDet = 0;
        
        /*RNG rng(12345);
         /// Draw contours
         Mat detected = Mat::zeros( resul.size(), CV_8UC3 );
         for( int i = 0; i< contours.size(); i++ )
         {
         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
         //drawContours( detected, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 1, Point() );
         drawContours( detected, contours, i, color, CV_FILLED, 8, hierarchy, 1, Point() );
         }//*/
        
        if (contours.size() == 0)
        {
            (*precisionBB)=0;
            (*recallBB)=0;
            return;
        }
        
        Mat matrix = Mat::zeros(1,contours.size(),CV_32FC1);
        
        FILE *fp;
        
        if( (fp = fopen(gtTXT.c_str(), "r+")) == NULL)
        {
            //printf("No such file\n");
            exit(1);
        }
        
        int pts[8]={0,0,0,0,0,0,0,0};
        //Mat gtRGB = Mat::zeros( resul.size(), CV_8UC3 );
        
        while (fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%*[^\n]",&pts[0], &pts[1], &pts[2], &pts[3],
                      &pts[4], &pts[5], &pts[6], &pts[7]) != EOF)//!feof(fp) )
        {
            
            Mat fila = Mat::zeros(1,contours.size(),CV_32FC1);
            Mat gt = Mat::zeros(resul.rows,resul.cols,CV_8UC1);
            
            Point points[1][4];
            points[0][0] = Point(pts[0],pts[1]);
            points[0][1] = Point(pts[2],pts[3]);
            points[0][2] = Point(pts[4],pts[5]);
            points[0][3] = Point(pts[6],pts[7]);
            
           /* printf("%s \n %d,%d,%d,%d,%d,%d,%d,%d,\n",gtTXT.c_str(),
                   pts[0], pts[1], pts[2], pts[3],
                   pts[4], pts[5], pts[6], pts[7]);getchar();//*/
            
            const Point* ppt[1] = { points[0] };
            int npt[] = { 4 };
            fillPoly( gt,
                     ppt,
                     npt,
                     1,
                     Scalar( 255, 255, 255 ),
                     CV_AA );
            
            
            
            /*fillPoly( gtRGB,
             ppt,
             npt,
             1,
             Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ),
             CV_AA );//*/
            
            numGT = numGT + 1;
            
            //printf("\n numGT %d:  ",numGT);
            
            for( int i = 0; i< contours.size(); i++ )
            {
                Mat detected = Mat::zeros( resul.size(), CV_8UC3 );
                drawContours( detected, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 1, Point() );
                cvtColor(detected,detected,CV_BGR2GRAY);
                /*imshow( "gt", gt );
                imshow( "Detected", detected );waitKey(0);//*/
                Mat inter,uni;
                bitwise_and(gt,detected,inter);
                //bitwise_or(gt,detected,uni);
                
                fila.at<float>(0,i) = float(countNonZero(inter))/float(countNonZero(gt));
            }
            
            if (numGT == 1)
                matrix = fila.clone();
            else
                vconcat(matrix, fila, matrix);
            
            
        
             
           /*  Mat imgTP;
             bitwise_and(gt,resul,imgTP);
             
             printf("size: %d AND: %d  area_overlap: %0.2f\n",countNonZero(gt),countNonZero(imgTP),float(countNonZero(imgTP))/(float)countNonZero(gt)*100.0);
             imshow("imgTP",imgTP);
             imshow("boundingbox",gt); waitKey(0);
            // */
            
        }
        
       // printf("\n\nnumGT: %d numDet: %d P_bb: %f R_bb: %f \n\n",numGT,numDet,(float)numDet/(float)contours.size(),(float)numDet/(float)numGT);
        
         printf("GT\t|");
         for(int c=0; c < matrix.cols; c++)
         {
         printf("\tbb %d",c);
         }
         printf("\n----------------------------\n");//*/
        
        Mat sumF = Mat::zeros( numGT, 1,CV_32FC1 );
        Mat sumC = Mat::zeros( 1, contours.size(),CV_32FC1 );
        
        for(int f=0; f < matrix.rows; f++)
        {
            float sum =0;
            for(int c=0; c < matrix.cols; c++)
                sum += matrix.at<float>(f,c);
            
            sumF.at<float>(f) = sum;
        }
        
        for(int c=0; c < matrix.cols; c++)
        {
            float sum =0;
            for(int f=0; f < matrix.rows; f++)
                sum += matrix.at<float>(f,c);
            
            sumC.at<float>(c) = sum;
        }
        
        
        
         for(int f=0; f < matrix.rows; f++)
         {
         printf("%d\t|",f);
         
         for(int c=0; c < matrix.cols; c++)
         {
         printf("\t%0.2f",matrix.at<float>(f,c));
         }
         
         //printf("\t|\t%0.2f",sumF.at<float>(f));
         
         printf("\n");
         }
         
         printf("\n----------------------------\n");//*/
        /*printf("\t|");
         for(int c=0; c < matrix.cols; c++)
         {
         printf("\t%0.2f",sumC.at<float>(c));
         }//*/
        
        int det = countNonZero((sumC > 0.5));
        int TP = countNonZero((sumF > 0.5));
        
        // printf("\ndet: %d TP: %d P_bb: %f R_bb: %f \n\n",det,TP,(float)det/(float)contours.size(),(float)TP/(float)numGT);
        
        (*precisionBB)=(float)det/(float)contours.size();
        (*recallBB)=(float)TP/(float)numGT;
        
        //imshow("show",gt);
       // waitKey(0);
        
        fclose (fp);
        
        
    }
    
    
};

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

