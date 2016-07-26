
//
//  main.cpp
//  test_superpixels
//
//  Created by Ana Cambra on 26/11/14.
//  Copyright (c) 2014 Ana Cambra. All rights reserved.
//

#include <iostream>

#include "superpixels.cpp"
//#include "utilsCaffe.cpp"
//#include <time.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <random>

#include "unitary_tests.cpp"

using namespace cv;

using namespace boost;

//using namespace boost::filesystem;
//#include <caffe/caffe.hpp>
//using namespace caffe;


#define LABEL_TEXT 1
#define LABEL_NOTEXT -1

//#define DEBUG_SP 1

int mLAB   = 0; int NBINS_L = 7; int NBINS_AB=7;
int mCONLAB   = 0;

int mRGB   = 0; int NBINS_RGB   = 256;
int mPEAKS = 0; int NBINS_PEAKS = 128;

int mLINES = 0; int NBINS_LINES = 128;

int mEDGES = 0; int NBINS_EDGES=8; int modeEDGES = 0;
int mEDDIR = 0; int NBINS_EDDIR = 8;
int mCONEDGES = 0;
int mCONEDDIR = 0;

int mHOG = 0; int NBINS_HOG=2916;
int mCONHOG = 0;


int mCAFFE = 0; string CAFFE_LAYER = "fc7"; int NUMCAFFE = 4096;
int mHYBRIDCNN = 0; string HYBRIDCNN_LAYER = "fc7"; int NUMHYBRIDCNN = 4096;
int mCONHYBRIDCNN = 0;

int mCONCAFFE = 0;
int mCONCAFFE2 = 0;

int mSEMANTIC = 0; int SEMANTIC_LABELS = 12; //SEGNET
int SEMANTIC2_LABELS = 23;
int SEMANTIC3_LABELS = 22;

int mCONTEXT = 0; // semantic_neigbourg
int mCONTEXT2 = 0; // oriented semantic_neigbourg


int mGLOBAL = 0; // oriented ALL semantic

bool DEBUG = false;

string DATASET = "svt1";//"ICDAR";

//caffe
utilsCaffe *_caffe;
utilsCaffe *_hybridcnn;

bool USE_PCA = true;
PCA* pca;

/*int nSAMPLES = 50;
 string nameSVM = "svm_LAB_PEAKS_" + to_string(nSAMPLES) + "_NOTZERO.xml";*/

Mat concatFileDescriptors(int id,Mat desRGB,Mat desLAB,Mat desCONLAB,
                          Mat desEDGES, Mat desCONEDGES,
                          Mat desEDDIR, Mat desCONEDDIR,
                          Mat desHOG, Mat desCONHOG,
                          Mat desCAFFE,
                          Mat desSEMANTIC, Mat desCONTEXT,Mat desCONTEXT2, Mat desGLOBAL,Mat desCONCAFFE,Mat desCONCAFFE2,
                          Mat desHYBRIDCNN, Mat desCONHYBRIDCNN)
{
    Mat des;
    
    if (mLAB != 0)
    {
        des = desLAB.row(id);
    }
    
    if (mCONLAB != 0)
    {
        if (des.rows != 0)
            hconcat(desCONLAB.row(id), des,des);
        else
            des=desCONLAB.row(id);
    }
    
    if (mRGB != 0)
    {
        if (des.rows != 0)
            hconcat(desRGB.row(id), des,des);
        else
            des=desRGB.row(id);
    }
    
    if (mEDGES != 0)
    {
        if (des.rows != 0)
            hconcat(desEDGES.row(id), des,des);
        else
            des=desEDGES.row(id);
    }
    
    if (mCONEDGES != 0)
    {
        if (des.rows != 0)
        hconcat(desCONEDGES.row(id), des,des);
        else
        des=desCONEDGES.row(id);
    }
    
    
    if (mEDDIR != 0)
    {
        if (des.rows != 0)
            hconcat(desEDDIR.row(id), des,des);
        else
            des=desEDDIR.row(id);
    }
    
    
    if (mCONEDDIR != 0)
    {
        if (des.rows != 0)
        hconcat(desCONEDDIR.row(id), des,des);
        else
        des=desCONEDDIR.row(id);
    }
    
    if (mHOG != 0)
    {
        if (des.rows != 0)
            hconcat(desHOG.row(id), des,des);
        else
            des=desHOG.row(id);
    }
    if (mCONHOG != 0)
    {
        if (des.rows != 0)
            hconcat(desCONHOG.row(id), des,des);
        else
            des=desCONHOG.row(id);
    }
    
    
    if (mSEMANTIC != 0)
    {
        if (des.rows != 0)
            hconcat(desSEMANTIC.row(id), des,des);
        else
            des=desSEMANTIC.row(id);
    }
    
    if (mCONTEXT != 0)
    {
        if (des.rows != 0)
            hconcat(desCONTEXT.row(id), des,des);
        else
            des=desCONTEXT.row(id);
        
        /*for (int i=0; i<des.cols; i++)
            printf("CONCAT %f  \n",des.at<float>(0,i));//*/
        
    }
    
    if (mCONTEXT2 != 0)
    {
        if (des.rows != 0)
            hconcat(desCONTEXT2.row(id), des,des);
        else
            des=desCONTEXT2.row(id);
        
        /*for (int i=0; i<des.cols; i++)
         printf("CONCAT %f  \n",des.at<float>(0,i));//*/
        
    }
    
    if (mGLOBAL != 0)
    {
        if (des.rows != 0)
            hconcat(desGLOBAL.row(id), des,des);
        else
            des=desGLOBAL.row(id);
        
        /*for (int i=0; i<des.cols; i++)
         printf("CONCAT %f  \n",des.at<float>(0,i));//*/
        
    }
    
    if (mCAFFE != 0)
    {
        if (des.rows != 0)
            hconcat(desCAFFE.row(id), des,des);
        else
            des=desCAFFE.row(id);
    }
    
    if (mCONCAFFE != 0)
    {
        if (des.rows != 0)
            hconcat(desCONCAFFE.row(id), des,des);
        else
            des=desCONCAFFE.row(id);
    }
    
    if (mCONCAFFE2 != 0)
    {
        if (des.rows != 0)
            hconcat(desCONCAFFE2.row(id), des,des);
        else
            des=desCONCAFFE2.row(id);
    }
    //HYBRIDCNN
    if (mHYBRIDCNN != 0)
    {
        if (des.rows != 0)
            hconcat(desHYBRIDCNN.row(id), des,des);
        else
            des=desHYBRIDCNN.row(id);
    }
    if (mCONHYBRIDCNN != 0)
    {
        if (des.rows != 0)
            hconcat(desCONHYBRIDCNN.row(id), des,des);
        else
            des=desCONHYBRIDCNN.row(id);
    }
    
    return des.clone();
}


//train SVM
//void descriptors2file(FILE *fout, float fid, Mat desID, float acc);

//bool file2descriptors(string file, int size, Mat *descriptors, Mat *accText);
bool descriptorFileText(string path,string img,int numID,
                        Mat *desRGB,
                        Mat *desLAB,  Mat *desCONLAB,
                        Mat *desEDGES, Mat *desCONEDGES,
                        Mat *desEDDIR, Mat *desCONEDDIR,
                        Mat *desHOG,  Mat *desCONHOG,
                        Mat *desCAFFE,
                        Mat *desSEMANTIC,
                        Mat *desCONTEXT,Mat *desCONTEXT2, Mat *desGLOBAL,
                        Mat *desCONCAFFE,  Mat *desCONCAFFE2,
                        Mat *desHYBRIDCNN, Mat *desCONHYBRIDCNN,
                        Mat *accText);

void saveSUPERPIXELSdescriptors(SuperPixels *SPCTE, string img, string nameEdges, string nameEdgesDIR, string nameSegmen);

void calculateSUPERPIXELSdescriptors(SuperPixels *SPCTE, string nameEdges, string nameEdgesDIR, string nameSegmen, Mat *descriptors, Mat *accText, string path, string img);

Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges = "", string nameEdgesDIR="", string nameSegmen = "");

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges, string dir_edgesDIR, string dir_segmen ,
                  int nSAMPLES, string nameSVM, string dir_des, string filename, string dataset);


//initialize structure of Superpixels
SuperPixels* svmSuperpixelsTEXT( string nameImage, int numLabels = 2, string imageGT = "", string nameSegmen = "")
{
    
    SuperPixels *SPCTE = new SuperPixels(nameImage);
    
    
    //init superpixels
    //SPCTE->initializeSuperpixels();
    
    
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    
    SPCTE->initializeSuperpixels();
    
    //TEXT LABELS
    SPCTE->setNUMLABELS(2);
    
    if (!imageGT.empty())
        SPCTE->initializeLabeling(imageGT, MODE_LABEL_NOTZERO);//MODE_LABEL_MEDIAN);//
    
    // if (mCAFFE == 1) SPCTE->initCaffe("/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel","/Users/acambra/Dropbox/test_caffe/deploy.prototxt");
    
    SPCTE->setNUMLABELS(numLabels);
    
    if (!nameSegmen.empty())
        SPCTE->initializeSegmentation(nameSegmen,SEMANTIC_LABELS,MODE_LABEL_NOTZERO,12);
    
   /* imshow("image",SPCTE->getImage());
    imshow("superpixels",SPCTE->getImageSuperpixels());
    waitKey(0);//*/
    
    return SPCTE;
    
}

void matGTfromFileBoundingBox(string dir_path, string dir_out = "")//name)//
{
    for (auto i = boost::filesystem::directory_iterator(dir_path); i != boost::filesystem::directory_iterator() ; i++)
    {
        if (!is_directory(i->path()))
        {
            //string IMAGE
            string file = i->path().filename().string();
            string name = dir_path + "/" + file;
            string nameImage = dir_out + "/" + file ;
            size_t lastindex = nameImage.find_last_of(".");
            nameImage = nameImage.substr(0, lastindex) + ".png";
            string extension = i->path().filename().extension().string();
            
            if (extension == ".txt")
            {
                Mat gt = Mat::zeros(720,1280,CV_8UC1);
                
                FILE *fp;
                
                if( (fp = fopen(name.c_str(), "r+")) == NULL)
                {
                    printf("No such file\n");
                    exit(1);
                }
                
                int pts[8]={0,0,0,0,0,0,0,0};
                
                while (fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%*[^\n]",&pts[0], &pts[1], &pts[2], &pts[3],
                              &pts[4], &pts[5], &pts[6], &pts[7]) != EOF)//!feof(fp) )
                {
                    
                    Point points[1][4];
                    points[0][0] = Point(pts[0],pts[1]);
                    points[0][1] = Point(pts[2],pts[3]);
                    points[0][2] = Point(pts[4],pts[5]);
                    points[0][3] = Point(pts[6],pts[7]);
                    
                    const Point* ppt[1] = { points[0] };
                    int npt[] = { 4 };
                    fillPoly( gt,
                             ppt,
                             npt,
                             1,
                             Scalar( 255, 255, 255 ),
                             CV_AA );
                    
                }
                //imshow("show",gt);waitKey(0);
                imwrite(nameImage,gt);
                fclose (fp);
                
            }//if image
        }//if file
    }//for*/
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

void cmpImageTXT(string gtTXT, string image, float* precisionBB, float *recallBB)
{
    //open file TXT
    
    Mat img = imread(image,CV_LOAD_IMAGE_COLOR);
     imshow("image",img);
    vector<Mat> bgr;
    split(img,bgr);
    
    Mat mask=(bgr[0] == 0);
    Mat resul=(bgr[1] == 0);
    
    bitwise_and(mask, resul, resul);
    mask = (bgr[2] == 255);
    bitwise_and(resul, mask, resul);
    imshow("resul",resul);
    
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
        (*precisionBB)=0;
        (*recallBB)=0;
        return;
    }
    
    int pts[8]={0,0,0,0,0,0,0,0};
    //Mat gtRGB = Mat::zeros( resul.size(), CV_8UC3 );
    
    while (fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%*[^\n]",&pts[0], &pts[1], &pts[2], &pts[3],
                  &pts[4], &pts[5], &pts[6], &pts[7]) != EOF)//!feof(fp) )
    {
        
        Mat fila = Mat::zeros(1,contours.size(),CV_32FC1);
        Mat gt = Mat::zeros(img.rows,img.cols,CV_8UC1);
        
        Point points[1][4];
        points[0][0] = Point(pts[0],pts[1]);
        points[0][1] = Point(pts[2],pts[3]);
        points[0][2] = Point(pts[4],pts[5]);
        points[0][3] = Point(pts[6],pts[7]);
        
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
                 CV_AA );*/
        
        numGT = numGT + 1;
        
        //printf("\n numGT %d:  ",numGT);
        
        for( int i = 0; i< contours.size(); i++ )
        {
            Mat detected = Mat::zeros( resul.size(), CV_8UC3 );
            drawContours( detected, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 1, Point() );
            cvtColor(detected,detected,CV_BGR2GRAY);
            //imshow( "Detected", detected );waitKey(0);
            Mat inter,uni;
            bitwise_and(gt,detected,inter);
            //bitwise_or(gt,detected,uni);
            
            fila.at<float>(0,i) = float(countNonZero(inter))/float(countNonZero(gt));
        }
        
        if (numGT == 1)
            matrix = fila.clone();
        else
            vconcat(matrix, fila, matrix);
        
        //imshow( "GT RBG", gtRGB); waitKey(0);
       
       /* Mat intersection =
        
        Mat imgTP;
        bitwise_and(gt,resul,imgTP);
        
        printf("size: %d AND: %d  area_overlap: %0.2f\n",countNonZero(gt),countNonZero(imgTP),float(countNonZero(imgTP))/(float)countNonZero(gt)*100.0);
        imshow("imgTP",imgTP);
        imshow("boundingbox",gt); waitKey(0);
        */
        
    }
    
    //printf("\n\nnumGT: %d numDet: %d P_bb: %f R_bb: %f \n\n",numGT,numDet,(float)numDet/(float)contours.size(),(float)numDet/(float)numGT);
    
   /* printf("GT\t|");
    for(int c=0; c < matrix.cols; c++)
    {
        printf("\tbb %d",c);
    }
    printf("\n----------------------------\n");*/
    
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
    
    
    
   /* for(int f=0; f < matrix.rows; f++)
    {
        printf("%d\t|",f);
        
        for(int c=0; c < matrix.cols; c++)
        {
            printf("\t%0.2f",matrix.at<float>(f,c));
        }
        
        //printf("\t|\t%0.2f",sumF.at<float>(f));
        
        printf("\n");
    }
    
    printf("\n----------------------------\n");*/
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
    
    //imshow("show",gt);waitKey(0);
   
    fclose (fp);
    
    
}

float cmpImage(string image, string gt, float* precision, float *recall)
{
    Mat img = imread(image,CV_LOAD_IMAGE_COLOR);
    // imshow("image",img);
    vector<Mat> bgr;
    split(img,bgr);
    
    Mat mask=(bgr[0] == 0);
    Mat resul=(bgr[1] == 0);
    
    bitwise_and(mask, resul, resul);
    mask = (bgr[2] == 255);
    bitwise_and(resul, mask, resul);
    
    
    //imshow("out",resul);
    
    Mat imgGT = imread(gt,CV_LOAD_IMAGE_GRAYSCALE);
    //imshow("gt",imgGT);waitKey(0);
    
    Mat imgTP,imgFP,imgFN;
    
    if (imgGT.rows != resul.rows || imgGT.cols != resul.rows)
       resize(imgGT, imgGT, resul.size());
    
    bitwise_and(imgGT,resul,imgTP);
    //imshow("imgTP",imgTP);
    
    Mat notGT;
    bitwise_not(imgGT,notGT);
    bitwise_and(notGT,resul,imgFP);
    
    Mat notresul;
    bitwise_not(resul,notresul);
    bitwise_and(imgGT,notresul,imgFN);
   // imshow("imgFP",imgFP);
   // waitKey(0);//*/
    
    int tp = countNonZero(imgTP);
    int tpfp = countNonZero(imgTP) + countNonZero(imgFP);
    int tpfn = countNonZero(imgTP) + countNonZero(imgFN);
    
    (*precision) = (float)tp / (float)tpfp;
    (*recall) = (float)tp / (float)tpfn ;
    
    return countNonZero(resul)*100.0/float(resul.cols*resul.rows);
    //printf("P:%f R:%f \n",(*precision),(*recall));

}

void parseTXT(string name, string out, string dir_gt,string dir_out)
{
    FILE *fout;
    
    fout = fopen(out.c_str(),"w");
    
    string search;
    ifstream inFile;
    string line;
    
    inFile.open(name);
    
    if(!inFile){
        cout << "Unable to open file" << endl;
        exit(1);
    }
    //cout << "Enter word to search for: ";
    //cin >>search;
    search = "TEST TIME";
    
    size_t pos;
    string title;
   
   // printf("CAFFE                                   85.177254           30.040001\n");
    fprintf(fout,"              OPTIONS                      time             percentage          precision             recall          precisionBB           recallBB            image\n");
    fprintf(fout,"------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    
    while(inFile.good())
    {
        getline(inFile,line); // get line from file
        pos=line.find(search); // search
        float percentage=0.0;
        char imgFile[300];
        
        if(pos!=string::npos) // string::npos is returned if string is not found
        {
            //cout <<"Found!: " << line << endl;;
            char line2[300];
            float time;
            
            sscanf(line.c_str(), " **** 	TEST TIME %s %f seconds",line2,&time);
            
            string line3;
            while(inFile.good())
            {
                getline(inFile,line3);
                pos=line3.find("TEXT CANDIDATES: ");
                int np;
                
                if(pos!=string::npos)
                {
                    sscanf(line3.c_str(), "**** 	TEXT CANDIDATES: %d (%f %%)\n",&np,&percentage);
                    getline(inFile,line3);
                    getline(inFile,line3);
                    sscanf(line3.c_str(), "\t* SVM solution saved: %s.png\n",imgFile);
                    break;
                }
            }
            
            ////////////
            // Pression / Recall
            size_t pos1,pos2;
             pos1=line3.find("/out/") + strlen("/out/");
            //pos1=line3.find("all_BB_response0/out/") + strlen("all_BB_response0/out/");
            //pos2=line3.find(".png__"); //cropped
            pos2=line3.find(".jpg");
            string file = line3.substr(pos1,(pos2-pos1));
            
            float precision,precisionBB=0.0;
            float recall,recallBB=0.0;
            //string resul = "/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Debug/" + string(imgFile);
            string resul = dir_out + string(imgFile);
            //string gt = "/Users/acambra/TESIS/datasets/svt1/train/gt/gt_" + file + ".png";
            string gt = dir_gt + file + ".png";
            percentage = cmpImage(resul, gt, &precision, &recall);
            
            //string gtTXT = "/Users/acambra/TESIS/datasets/svt1/train/gt/gt_" + file + ".txt";
            string gtTXT = dir_gt + file + ".txt";
            //cmpImageTXT(gtTXT,resul, &precisionBB, &recallBB);
            
            
            fprintf(fout,"%-40s%-20f%-20f%-20f%-20f%-20f%-20f%-20s\n",line2,time,percentage,precision,recall,precisionBB,recallBB,file.c_str());//getchar();
            
            // getline(inFile,line2);
            //
            //break;
        }
    }
    fclose(fout);
}

void parseDIR(string dir,string folder,string out)
{
    FILE *fout;
    
    fout = fopen(out.c_str(),"w");
    
    
    string dir_path = dir + "/" + folder;
    
    fprintf(fout," image-SegNet      percentage          precision             recall\n");
    fprintf(fout,"------------------------------------------------------------------------\n");
    
    for (auto i = boost::filesystem::directory_iterator(dir_path); i != boost::filesystem::directory_iterator(); i++)
    {
        if (!is_directory(i->path()))
        {
            string nameImage = dir_path + "/" + i->path().filename().string();
            string extension = i->path().filename().extension().string();
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                size_t pos1,pos2;
                pos1=0;//i->path().filename().string().find("img_");
                pos2=i->path().filename().string().find(".jpg");//find("_SegNet");
                
                string file = i->path().filename().string().substr(pos1,(pos2-pos1));
                string gt =  dir + "/gt/gt_" + file + ".png";
                
               // gt.replace(gt.find(".jpg"), sizeof(".jpg")-1, ".png");
                
                Mat imgGT = imread(gt,CV_LOAD_IMAGE_GRAYSCALE);
                
                Mat resul = imread(nameImage,CV_LOAD_IMAGE_GRAYSCALE);
                
                
                Mat imgTP,imgFP,imgFN;
                
                if (imgGT.rows != resul.rows || imgGT.cols != resul.rows)
                    resize(resul, resul, imgGT.size());
                
                bitwise_and(imgGT,resul,imgTP);
                
                Mat notGT;
                bitwise_not(imgGT,notGT);
                bitwise_and(notGT,resul,imgFP);
                
                Mat notresul;
                bitwise_not(resul,notresul);
                bitwise_and(imgGT,notresul,imgFN);
                
                int tp = countNonZero(imgTP);
                int tpfp = countNonZero(imgTP) + countNonZero(imgFP);
                int tpfn = countNonZero(imgTP) + countNonZero(imgFN);
                
                float precision = (float)tp / (float)tpfp;
                float recall = (float)tp / (float)tpfn ;
                
                float per= countNonZero(resul)*100.0/float(resul.cols*resul.rows);
                
                float precisionBB,recallBB;
                string gtTXT =  gt.replace(gt.find(".png"), sizeof(".png")-1, ".txt");
                //UnitaryTests::cmpImageTXT(gtTXT,resul, &precisionBB, &recallBB);
                
                fprintf(fout,"%-20s%-20f%-20f%-20f%-20f%-20f\n",file.c_str(),per,precision,recall,precisionBB, recallBB);
               // printf("---> %s \n %s  %f %f %f n\n",nameImage.c_str(),gt.c_str(),precision, recall, per);
               // imshow("GT",imgGT);imshow("resul",resul); waitKey(0);
            }
        }
    }
    
    fclose(fout);
}

/********************************************************************************************************************************
 
    MAIN
 
 ********************************************************************************************************************************/

int main(int argc, const char * argv[]) {
    
   /* matGTfromFileBoundingBox("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_localization_transcription_gt/us-ascii/" //gt_img_1.txt"//);
    ,
                            "/Users/acambra/Dropbox/dataset/ICDAR/ch4_images_gt");
    return 1;//*/
   
    //tp pn nn np
    
    /*float precision, recall;
    cmpImageTXT("/Users/acambra/TESIS/datasets/svt1/train/gt/gt_img_3.txt",
                "/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Debug/out/img_3.jpg_CAFFE_SEMANTIC_570.png",
                &precision,&recall);
    
    return 0;//*/
    
    //parseDIR("/Users/acambra/SegNet/SegNet_text/ICDAR/test","train_svt1_iter_60000","/Users/acambra/SegNet/SegNet_text/ICDAR/test/test_SegNet_svt1_60000_2.txt");
    /*parseDIR("/Users/acambra/TESIS/datasets/Pascal_context/test",
             "doors_iter_60000",
             "/Users/acambra/TESIS/datasets/Pascal_context/test/test_SegNet_doors.txt");
    return 2;//*/
    
    parseTXT("/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Release/stdout_doors_test_233_combined_response0_3.txt",
             "/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Release/resul_doors_test_233_combined_response0_3.txt",
             "/Users/acambra/TESIS/datasets/pascal_context/test/gt/gt_", "");
    return 1;//*/
   /* parseTXT("/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Release/stdout_svt1_test_low_response0_2.txt",
     "/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Release/resul_svt1_test_low_response0_2.txt",
     "/Users/acambra/TESIS/datasets/svt1/test/gt/gt_", "");
     return 1;//*/
    

    //parse argv
    float timeDescriptors=0;
    string nameSVM ="";
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("svmOptions", boost::program_options::value<std::string>()->required(), "descriptors train")
    ("v", "debug mode ON")
    ("svmTest", boost::program_options::value<std::string>()->default_value("")->implicit_value(""),"test svm")
    ("image", boost::program_options::value<std::string>(), "image")
    ("labeled", boost::program_options::value<std::string>(), "image GT labeled")
    ("edges", boost::program_options::value<std::string>(), "path image edges")
    ("eddir", boost::program_options::value<std::string>(), "path image normals/orientation edges")
    ("semantic", boost::program_options::value<std::string>(), "path image semantic segmentation")
    ("numLabels", boost::program_options::value<int>()->default_value(2), "num labels")
    ("svmTrain",boost::program_options::value<std::string>()->default_value("")->implicit_value(""),"train svm")
    ("dirout", boost::program_options::value<std::string>(), "dir images out")
    ("nSamples", boost::program_options::value<int>()->required(), "test svm")
    ("dataset", boost::program_options::value<std::string>()->default_value("svt1"), "train dataset")
    ("help", "produce help message");
    
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc),parameters);
    //  boost::program_options::notify(parameters);
    
    
    //if (parameters.find("help") != parameters.end()) {
    if (parameters.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    
    if (parameters.find("v") != parameters.end()) {
        DEBUG = true;
    }
    
    //SVM Options
    int nSAMPLES = parameters["nSamples"].as<int>(); printf("nsamples \n");
    
    if (parameters["svmOptions"].as<std::string>().find("CONLAB7") != std::string::npos)
    {
        mCONLAB = 1;
        NBINS_L = 7;
        NBINS_AB= 7;
        
    }
    
    if (parameters["svmOptions"].as<std::string>().find("_LAB7") != std::string::npos)
    {
        mLAB = 1;
        NBINS_L = 7;
        NBINS_AB= 7;
        
    }else if (parameters["svmOptions"].as<std::string>().find("LAB9") != std::string::npos)
    {
        mLAB = 1;
        NBINS_L = 9;
        NBINS_AB= 9;
        
    }else if (parameters["svmOptions"].as<std::string>().find("_LAB") != std::string::npos)
    {
        mLAB = 1;
    }
    
    if (parameters["svmOptions"].as<std::string>().find("RGB") != std::string::npos)
        mRGB = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("PEAKS") != std::string::npos)
        mPEAKS = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("CONEDDIR") != std::string::npos)
        mCONEDDIR = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("_EDDIR") != std::string::npos)
        mEDDIR = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("CONEDGES") != std::string::npos)
    {
        mCONEDGES = 1;
        
        if (modeEDGES == 0) //hist
        NBINS_EDGES = 8;
        else if (modeEDGES == 1) //moments
        NBINS_EDGES = 4;
        else if (modeEDGES == 2)
        NBINS_EDGES = 8 + 4;
        
    }
    
    if (parameters["svmOptions"].as<std::string>().find("_EDGES") != std::string::npos)
    {
        mEDGES = 1;
        
        if (modeEDGES == 0) //hist
            NBINS_EDGES = 8;
        else if (modeEDGES == 1) //moments
            NBINS_EDGES = 4;
        else if (modeEDGES == 2)
            NBINS_EDGES = 8 + 4;
        
    }
    
    if (parameters["svmOptions"].as<std::string>().find("_HOG") != std::string::npos)
    {
        mHOG = 1;
    }
    
    if (parameters["svmOptions"].as<std::string>().find("_CONHOG") != std::string::npos)
    {
        mCONHOG = 1;
    }
    
    if (parameters["svmOptions"].as<std::string>().find("SEMANTIC2") != std::string::npos)
    {
        mSEMANTIC = 1; SEMANTIC_LABELS = SEMANTIC2_LABELS;
    }
    else  if (parameters["svmOptions"].as<std::string>().find("SEMANTIC3") != std::string::npos)
    {
        mSEMANTIC = 1; SEMANTIC_LABELS = SEMANTIC3_LABELS;
    } else if (parameters["svmOptions"].as<std::string>().find("_SEMANTIC") != std::string::npos)
    {
        mSEMANTIC = 1;
    }
    
    if (parameters["svmOptions"].as<std::string>().find("CONTEXT2") != std::string::npos)
        mCONTEXT2 = 1;
    else if (parameters["svmOptions"].as<std::string>().find("CONTEXT") != std::string::npos)
        mCONTEXT = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("GLOBAL") != std::string::npos)
    {
        mGLOBAL = 1;
    }
    

    string svmType="";
     if (parameters["svmTrain"].as<std::string>().length() > 0)
         svmType = parameters["svmTrain"].as<std::string>();
     if (parameters["svmTest"].as<std::string>().length() > 0)
            svmType= parameters["svmTest"].as<std::string>();
    
    if (svmType.length()>0)
        svmType =  "_" + svmType;
    
    string dataset = parameters["dataset"].as<std::string>();    
    printf("TRAIN Dataset: %s\n",dataset.c_str());
    
    nameSVM = string("svm_") + parameters["svmOptions"].as<std::string>() + string("_") + to_string(nSAMPLES) + svmType + string("_NOTZERO.xml");
    string nameSVM2 = string("/Users/acambra/TESIS/datasets/"+ dataset +"/train/BB_descriptors/") + nameSVM;
    
    // printf("SVM %s. %s.\n", nameSVM.c_str(),nameSVM2.c_str() ); getchar();
    
    
    
    ////////////////////////
    //SVM train o load file
     ////////////////////////
    
    if (parameters.count("svmTrain")) {
        
        string svmType= parameters["svmTrain"].as<std::string>();
        printf("TYPE: %s\n",svmType.c_str());//getchar();
        printf("NAME SVM FILE: %s\n",nameSVM2.c_str());//getchar();
        
        printf("==================================================\n");
        printf("**** \tTRAIN SVM_%s numSamples %d\n",parameters["svmOptions"].as<std::string>().c_str(),nSAMPLES);
        printf("==================================================\n");
        
        ifstream infile(nameSVM2);
        if (! infile.good()) {
            
            if (parameters["svmOptions"].as<std::string>().find("_CONCAFFE2") != std::string::npos)
            {
                mCONCAFFE2 = 1;
                //initCaffe
                /*string model = "/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel";
                string proto = "/Users/acambra/Dropbox/test_caffe/deploy.prototxt" ;
                //printf("Model CAFFE: %s PROTO: %s\n",model.c_str(),proto.c_str());getchar();
                
                _caffe = new utilsCaffe(model,proto);*/
            }else if (parameters["svmOptions"].as<std::string>().find("CONCAFFE") != std::string::npos)
            {
                mCONCAFFE = 1;
            }
            
            if (parameters["svmOptions"].as<std::string>().find("_CAFFE") != std::string::npos)
            {
                mCAFFE = 1;
            }
            
            if (parameters["svmOptions"].as<std::string>().find("_HYBRIDCNN") != std::string::npos)
            {
                mHYBRIDCNN = 1;
            }
            if (parameters["svmOptions"].as<std::string>().find("_CONHYBRIDCNN") != std::string::npos)
            {
                mCONHYBRIDCNN = 1;
            }
            
            //default
            string path = "/Users/acambra/TESIS/datasets/";// "/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Debug/";
            
            string svmOptions = dataset + string("/train/svm_") + parameters["svmOptions"].as<std::string>() + string("_") + to_string(nSAMPLES) + string("_NOTZERO.xml_des.yaml");
            
            trainSVMText(path + dataset + "/train/images",//"ICDAR/train/images",
                         path + dataset + "/train/gt",//"ICDAR/train/gt",
                         path + dataset + "/train/edges",//"ICDAR/train/edges",
                         path + dataset + "/train/normals",//"ICDAR/train/normals",
                        /* path + "ICDAR/train/semantic",*/
                         path + dataset + "/train/semantic",//"/Users/acambra/SegNet/ICDAR-SegNet",
                         nSAMPLES,
                         nameSVM2,
                         path + dataset + "/train/BB_descriptors",//"ICDAR/train/descriptors",
                         path + svmOptions,
                         dataset);
           
           /* path + "ICDAR/test/images",
            "",
            path + "ICDAR/test/edges",
            path + "ICDAR/test/normals",
            "/Users/acambra/SegNet/ICDAR-SegNet",
            nSAMPLES,
            nameSVM2,
            path + "train/ICDAR/descriptors");*/
            
            if (_caffe) delete _caffe;
            
            printf("\n\t* SVM save: %s\n\n", nameSVM2.c_str());
            
        }
        else{
            printf("\n\t* SVM Load: %s\n\n", nameSVM2.c_str());
        }
        
    }//parameters.find("svmTrain")*/
    
    ////////////////////////
    //evaluate test SVM
    ////////////////////////
    
    
    
    
   // if (parameters.find("svmTest") != parameters.end() && parameters["svmTest"].as<std::string>().length() > 0) {
    if (parameters.count("svmTest")) {
        
        string dir_out="";
        
        if (parameters.find("dirout") != parameters.end())
        {
            dir_out = parameters["dirout"].as<std::string>();
        }
        else
        {
            printf("Error dirout not found!\n");
            return -1;
        }
        
        CvSVM SVM;
        SVM.load(nameSVM2.c_str());
        
        boost::filesystem::path inputImage(parameters["image"].as<std::string>());
        
        ////////////////////////
        //SUPERPIXELS
        
        
        string nameImage = inputImage.string();
        string nameGT="";
       
        string nameSegmen ="";
        if (mSEMANTIC == 1 || mCONTEXT == 1 || mCONTEXT2 == 1 || mGLOBAL)
            nameSegmen = parameters["semantic"].as<std::string>();
        
        SuperPixels *SPCTE;
        
        SPCTE= svmSuperpixelsTEXT(nameImage,2,nameGT,nameSegmen);
        
        Mat imgSP = SPCTE->getImage().clone();
        char k=-1;
        
        string nameWindow = "Superpixel ";
        
        //test
        
        if (parameters.find("labeled") != parameters.end())
            nameGT = parameters["labeled"].as<std::string>();
        
        int numText=0;
        int numNOText=0;
        clock_t start = clock();
        
        if (parameters["svmOptions"].as<std::string>().find("_CONCAFFE2") != std::string::npos)
        {
            mCONCAFFE2 = 1;
        }
        
        if (parameters["svmOptions"].as<std::string>().find("_CAFFE") != std::string::npos)
        {
            mCAFFE = 1;
        }
        //HYBRIDCNN
        if (parameters["svmOptions"].as<std::string>().find("_HYBRIDCNN") != std::string::npos)
        {
            mHYBRIDCNN = 1;
        }
        if (parameters["svmOptions"].as<std::string>().find("_CONHYBRIDCNN") != std::string::npos)
        {
            mCONHYBRIDCNN = 1;
        }
        
        //READ????
        Mat desRGB,desLAB,desCONLAB,desEDGES,desEDDIR,desCONEDGES,desCONEDDIR,desHOG,desCONHOG,desCAFFE,desCONCAFFE,desCONCAFFE2,desSEMANTIC,desCONTEXT,desCONTEXT2,desGLOBAL,desHYBRIDCNN,desCONHYBRIDCNN,accText;
        
        int num = SPCTE->maxID+1;
        string file = nameImage.substr((int)nameImage.find_last_of("/")+1);
        
        string dir_des=nameImage.substr(0,(int)nameImage.find_last_of("/"));
        dir_des=dir_des.substr(0,(int)dir_des.find_last_of("/"))+ "/BB_descriptors/";
        
       // desSEMANTIC = Mat::zeros(num,2*(SEMANTIC_LABELS),CV_32FC1);*/
        bool readDescriptor = descriptorFileText(dir_des,file, num,
                                                 &desRGB,
                                                 &desLAB,&desCONLAB,
                                                 &desEDGES,&desCONEDGES,
                                                 &desEDDIR,&desCONEDDIR,
                                                 &desHOG, &desCONHOG,
                                                 &desCAFFE,
                                                 &desSEMANTIC,
                                                 &desCONTEXT,&desCONTEXT2,&desGLOBAL,
                                                 &desCONCAFFE,&desCONCAFFE2,
                                                 &desHYBRIDCNN,&desCONHYBRIDCNN,
                                                 &accText);
        
        string nameEdges="";
        if (parameters.find("edges") != parameters.end())
            nameEdges = parameters["edges"].as<std::string>();
        
        string nameEdgesDIR="";
        if (parameters.find("eddir") != parameters.end())
            nameEdgesDIR = parameters["eddir"].as<std::string>();
        
        
        if  (!readDescriptor && (mCAFFE == 1 || mCONCAFFE == 1 || mCONCAFFE2 == 1))
        {
            string model = "/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel";
            string proto = "/Users/acambra/Dropbox/test_caffe/deploy.prototxt" ;
            // printf("Model CAFFE: %s PROTO: %s\n",model.c_str(),proto.c_str());getchar();
            
            try
            {
                _caffe = new utilsCaffe(model,proto);
            }
            catch (int e)
            {
                printf("ERROR in LOAD CAFFE: %d\n", e);
            }
        }
        //HYBRIDCNN
        if  (!readDescriptor && (mHYBRIDCNN == 1 || mCONHYBRIDCNN == 1))
        {
            try
            {
                _hybridcnn = new utilsCaffe("/Users/acambra/TESIS/caffeModels/hybridCNN/hybridCNN_iter_700000.caffemodel",
                                    "/Users/acambra/TESIS/caffeModels/hybridCNN/hybridCNN_deploy.prototxt");
            }
            catch (int e)
            {
                printf("ERROR in LOAD HYBRIDCNN: %d\n", e);
            }
        }
        
        //saveDescriptors
        if (!readDescriptor)
        {
            
            clock_t ini = clock();
            //save
            saveSUPERPIXELSdescriptors(SPCTE,dir_des + file,nameEdges,nameEdgesDIR,nameSegmen);
            //printf("\t\t\t* Time File descriptors %f \n\n",(float) (((double)(clock() - ini)) / CLOCKS_PER_SEC));
            timeDescriptors = (float) (((double)(clock() - ini)) / CLOCKS_PER_SEC);
            
            
            readDescriptor = descriptorFileText(dir_des,file, num,
                                                &desRGB,
                                                &desLAB,&desCONLAB,
                                                &desEDGES,&desCONEDGES,
                                                &desEDDIR,&desCONEDDIR,
                                                &desHOG, &desCONHOG,
                                                &desCAFFE,
                                                &desSEMANTIC,
                                                &desCONTEXT,&desCONTEXT2,&desGLOBAL,
                                                &desCONCAFFE,&desCONCAFFE2,
                                                &desHYBRIDCNN,&desCONHYBRIDCNN,
                                                &accText);//*/

        }
        
        Mat desID,desF;
        
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
            
            //DESCRIPTOR SUPERPIXEL: calculate or read
           
            desID.release();//,desF;
            
            
            /*desF =  concatFileDescriptors(id,desRGB,desLAB,desCONLAB,
                                           desEDGES, desCONEDGES,
                                           desEDDIR, desCONEDDIR,
                                           desHOG, desCONHOG,
                                           desCAFFE, desSEMANTIC, desCONTEXT,desCONTEXT2,desGLOBAL,desCONCAFFE,desCONCAFFE2);//*/
            
            if (readDescriptor)
                desID =  concatFileDescriptors(id,desRGB,desLAB,desCONLAB,
                                           desEDGES, desCONEDGES,
                                           desEDDIR, desCONEDDIR,
                                           desHOG, desCONHOG,
                                           desCAFFE, desSEMANTIC, desCONTEXT,desCONTEXT2,desGLOBAL,desCONCAFFE,desCONCAFFE2,
                                           desHYBRIDCNN,desCONHYBRIDCNN);
            else //*/
                desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen).clone();

        //check file and calculated
           /*for (int i=0; i<desID.cols; i++)
           {
               
               double leido = (double)desF.at<float>(0,i);
               double calculado = (double)desID.at<float>(0,i);
                if (leido != calculado )
                    printf("ERROR!!!!!!!!!!! id %d calculado: %0.8f leido: %0.8f  \n",id,calculado,leido);
           }
            //*/
            
            //evaluate SVM
            //float threshold = 0.5;
            if (SVM.get_var_count() != 0)
            {
                
                    float response = SVM.predict(desID);
                    printf("RESPONSE SVM id: %d  %f\n",id,response);
                    //paint image with SVM response
                    if (response == (LABEL_TEXT))
                    {
                        response = SVM.predict(desID,true);//distance
                        imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();
                        //imgSP_2 = SPCTE->paintSuperpixel(imgSP_2, id,new Scalar(0,255,0)).clone();
                        
                        numText += SPCTE->numPixels(id);
                        if (DEBUG) printf("\t*** SVM response id: %d  TEXT (%f)\n",id,response);
                    }
                    else
                    {
                        response = SVM.predict(desID,true); //distance
                       /* if (response <= 0.0)
                        {
                        //    imgSP_2 = SPCTE->paintSuperpixel(imgSP_2, id, new Scalar(0,255,0)).clone();
                            imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();
                            if (DEBUG) printf("\t*** SVM response id: %d  TEXT????? (%f)\n",id,response);
                        }
                        else
                        {*/
                            numNOText += SPCTE->numPixels(id);
                            if (DEBUG) printf("\t*** SVM response id: %d  NO TEXT (%f)\n",id,response);
                       // }
                            
                    }
               // }
            }
            else
            {
                if (DEBUG) printf("** ERROR: SVM not found: %s \n",nameSVM.c_str());
            }
            
            if (DEBUG == 1) {
                
                destroyWindow(nameWindow.c_str());
                nameWindow = "Superpixel " + to_string(id);
                imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));
                
                if ((char)k != 'c')
                    k = waitKey(0);
            }
            
           // desID.release();
            
           // printf(".");
            
        }//for id SPCTE//*/
        
        //if (_caffe) delete _caffe;
        
        printf("\n===================================================================\n");
        printf("**** \tTEST TIME %s %f seconds\n",parameters["svmOptions"].as<std::string>().c_str(),(float) (((double)(clock() - start)) / CLOCKS_PER_SEC));
        printf("===================================================================\n");
        printf("**** Superpixels: %f seconds\n ",SPCTE->timeSuperpixels);
        printf("**** Descriptors: %f seconds\n ",timeDescriptors);
        /*if (mLAB == 1)      printf("**** LAB: %f seconds\n ",SPCTE->timeLAB);
        if (mRGB == 1)      printf("**** RGB: %f seconds\n ",SPCTE->timeRGB);
        if (mEDGES == 1)    printf("**** EDGES: %f seconds\n ",SPCTE->timeEDGES);
        if (mEDDIR == 1)    printf("**** EDDIR: %f seconds\n ",SPCTE->timeEDGESDIR);
        if (mCAFFE == 1)    printf("**** CAFFE: %f seconds\n ",SPCTE->timeCAFFE);
        if (mSEMANTIC == 1) printf("**** SEMANTIC: %f seconds\n ",SPCTE->timeSEMANTIC);//*/
        
        printf("-------------------------------------------------------------------\n");
        printf("**** \tTEXT CANDIDATES: %d (%0.2f %%)\n", numText,
               100.0 * (float)numText / (float)(SPCTE->getImage().rows*SPCTE->getImage().cols));
        
        if (DEBUG == 1)
        {
            imshow(nameSVM,imgSP);
           // imshow("0.5 distance from border SVM",imgSP_2);
            waitKey(0);
        }
        else{
            
            size_t found1 = nameImage.find_last_of("/");
            string name = dir_out + "out/"+ nameImage.substr(found1+1) + "_" + parameters["svmOptions"].as<std::string>()+ "_" + to_string(nSAMPLES)+ svmType +".png";
            imwrite(name,imgSP);
            printf("\n\t* SVM solution saved: %s\n",name.c_str());
        }
        
        imgSP.release();
        delete SPCTE;
        
        printf("===================================================================\n");
        
    }//svmTest
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TRAIN SVM
//change!!!!!!
bool descriptorFileText(string path, string img,int numID,
                        Mat *desRGB,
                        Mat *desLAB, Mat *desCONLAB,
                        Mat *desEDGES, Mat *desCONEDGES,
                        Mat *desEDDIR,  Mat *desCONEDDIR,
                        Mat *desHOG, Mat *desCONHOG,
                        Mat *desCAFFE, Mat *desSEMANTIC,
                        Mat *desCONTEXT,Mat *desCONTEXT2, Mat *desGLOBAL,
                        Mat *desCONCAFFE, Mat *desCONCAFFE2,
                        Mat *desHYBRIDCNN, Mat *desCONHYBRIDCNN,
                        Mat *accText)
{
    //check Files
    img = path + img;
    
    string fRGB = img + "_RGB" + ".bin";
    string fLAB = img + "_LAB" + ".bin";
    
    if (NBINS_L == 7 && NBINS_AB == 7)
        fLAB = img + "_LAB7" + ".bin";
    else if (NBINS_L == 9 && NBINS_AB == 9)
        fLAB = img + "_LAB9" + ".bin";
    
    string fCONLAB = img + "_CONLAB7" + ".bin";
    
    string fEDGES = img + "_EDGES" + ".bin";
    string fEDDIR = img + "_EDDIR" + ".bin";
    string fCONEDGES = img + "_CONEDGES" + ".bin";
    string fCONEDDIR = img + "_CONEDDIR" + ".bin";
    
    string fHOG = img + "_HOG" + ".bin";
    string fCONHOG = img + "_CONHOG" + ".bin";
    
    string fCAFFE = img + "_CAFFE" + ".bin";
    string fCONCAFFE = img + "_CONCAFFE" + ".bin";
    string fCONCAFFE2 = img + "_CONCAFFE2" + ".bin";
    string fSEMANTIC = img + "_SEMANTIC" + ".bin";
    string fCONTEXT = img + "_CONTEXT" + ".bin";
    string fCONTEXT2 = img + "_CONTEXT2" + ".bin";
    string fGLOBAL = img + "_GLOBAL" + ".bin";
    
    string fHYBRIDCNN = img + "_HYBRIDCNN" + ".bin";
    string fCONHYBRIDCNN = img + "_CONHYBRIDCNN" + ".bin";
    
    int numDES = 0;
    
    (* accText) = Mat::zeros(1, numID, CV_32FC1);
    
    bool ok = true;

    
    if (mRGB == 1 && ok)
    {
        numDES = (3*NBINS_RGB);
        (*desRGB) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fRGB,numDES,desRGB, accText);
    }
    
    if (mLAB == 1 && ok)
    {
        numDES =  NBINS_L + (2*NBINS_AB);
        (*desLAB) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fLAB,numDES,desLAB, accText);
    }
    
    if (mCONLAB == 1 && ok)
    {
        numDES =  NBINS_L + (2*NBINS_AB);
        (*desCONLAB) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONLAB,numDES,desCONLAB, accText);
    }
    
    if (mEDGES == 1 && ok)
    {
        numDES = NBINS_EDGES;
        (*desEDGES) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fEDGES,numDES,desEDGES, accText);
    }
    
    if (mCONEDGES == 1 && ok)
    {
        numDES = NBINS_EDGES;
        (*desCONEDGES) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONEDGES,numDES,desCONEDGES, accText);
    }
    
    if (mEDDIR == 1 && ok)
    {
        numDES = NBINS_EDDIR;
        (*desEDDIR) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fEDDIR,numDES,desEDDIR, accText);
    }
    
    if (mCONEDDIR == 1 && ok)
    {
        numDES = NBINS_EDDIR;
        (*desCONEDDIR) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONEDDIR,numDES,desCONEDDIR, accText);
    }
    
    if (mHOG == 1 && ok)
    {
        numDES =  NBINS_HOG;
        (*desHOG) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fHOG,numDES,desHOG, accText);
    }
    
    if (mCONHOG == 1 && ok)
    {
        numDES =  NBINS_HOG;
        (*desCONHOG) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONHOG,numDES,desCONHOG, accText);
    }

    
    if (mCAFFE == 1 && ok)
    {
        numDES = NUMCAFFE;
        (*desCAFFE) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCAFFE,numDES,desCAFFE, accText);
    }
    
    if (mCONCAFFE == 1 && ok)
    {
        numDES = NUMCAFFE;
        (*desCONCAFFE) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONCAFFE,numDES,desCONCAFFE, accText);
    }
    
    if (mCONCAFFE2 == 1 && ok)
    {
        numDES = NUMCAFFE;
        (*desCONCAFFE2) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONCAFFE2,numDES,desCONCAFFE2, accText);
    }
    
    if (mSEMANTIC == 1 && ok)
    {
        numDES = (SEMANTIC_LABELS);
        (*desSEMANTIC) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fSEMANTIC,numDES,desSEMANTIC,accText);
    }
    
    if (mCONTEXT == 1 && ok)
    {
        numDES = (SEMANTIC_LABELS);
        (*desCONTEXT) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONTEXT,numDES,desCONTEXT,accText);

    }
    
    if (mCONTEXT2 == 1 && ok)
    {
        numDES = (4*SEMANTIC_LABELS);
        (*desCONTEXT2) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONTEXT2,numDES,desCONTEXT2,accText);
    }
    
    if (mGLOBAL == 1 && ok)
    {
        numDES = (4*SEMANTIC_LABELS);
        (*desGLOBAL) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fGLOBAL,numDES,desGLOBAL,accText);
    }
    //HYBRIDCCN
    if (mHYBRIDCNN == 1 && ok)
    {
        numDES = NUMHYBRIDCNN;
        (*desHYBRIDCNN) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fHYBRIDCNN,numDES,desHYBRIDCNN, accText);
    }
    if (mCONHYBRIDCNN == 1 && ok)
    {
        numDES = NUMHYBRIDCNN;
        (*desCONHYBRIDCNN) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && Descriptors::file2descriptors(fCONHYBRIDCNN,numDES,desCONHYBRIDCNN, accText);
    }
    
    return ok;
}


Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges,string nameEdgesDIR, string nameSegmen)
{
    //show info
    
    Mat des;
    
    if (DEBUG)
    {
        string info= "";
        if (mLAB != 0)
            info += "LAB ";
        
        if (mCONLAB != 0)
            info += "CONLAB ";
        
        if (mRGB != 0)
            info += "RGB ";
        
        if (mPEAKS != 0)
            info += "PEAKS ";
        
        if (mEDGES != 0)
            info += "EDGES  ";
        
        if (mCONEDGES != 0)
            info += "CONEDGES  ";
        
        if (mEDDIR != 0)
            info += "EDDIR  ";
        if (mCONEDDIR != 0)
            info += "CONEDDIR  ";
        
        if (mHOG != 0)
            info += "HOG  ";
        if (mCONHOG != 0)
            info += "CONHOG  ";
        
        if (mCAFFE != 0)
            info += "CAFFE ";
        
        if (mCONCAFFE != 0)
            info += "CONCAFFE ";
        
        if (mCONCAFFE2 != 0)
            info += "CONCAFFE2 ";
        
        if (mSEMANTIC != 0)
            info += "SEMANTIC ";
        
        if (mCONTEXT != 0)
            info += "CONTEXT ";
        
        if (mCONTEXT2 != 0)
            info += "CONTEXT2 ";
        
        if (mGLOBAL != 0)
            info += "GLOBAL ";
        
        if (mHYBRIDCNN !=0) {
            info += "HYBRIDCNN ";
        }
        
        if (mCONHYBRIDCNN !=0) {
            info += "CONHYBRIDCNN ";
        }
        
        printf("descriptors: %s id %d/%d \n",info.c_str(),id,SPCTE->maxID);
    }
    
    //EDGES
    Mat imgEdges = Mat();
    if (mEDGES == 1 || mCONEDGES ==1)
    {
        // string nameEdges = dir_edges + "/" + i->path().filename().string();
        // nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
        imgEdges = imread(nameEdges,CV_LOAD_IMAGE_COLOR);
        if (imgEdges.data == NULL)
        {
            printf("ERROR: No edges image found\n");
        }
    }
    
    Mat imgEdgesDIR = Mat();
    if (mEDDIR == 1 || mCONEDDIR ==1)
    {
        imgEdgesDIR = loadMatFromYML(nameEdgesDIR,"N");
        if (imgEdgesDIR.data == NULL)
        {
            printf("ERROR: No orientation edges image found\n %s\n",nameEdgesDIR.c_str());
        }
        
        imgEdges = imread(nameEdges,CV_LOAD_IMAGE_COLOR);
        if (imgEdges.data == NULL)
        {
            printf("ERROR: No edges image found\n%s\n",nameEdges.c_str());
        }
    }

    
    
    SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                          ,mLAB,NBINS_L,NBINS_AB \
                                          ,mRGB,NBINS_RGB\
                                          ,mPEAKS,NBINS_PEAKS \
                                          ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                          ,mEDDIR,NBINS_EDDIR,imgEdgesDIR\
                                          ,0,CAFFE_LAYER,NUMCAFFE\
                                          ,mSEMANTIC,SEMANTIC_LABELS \
                                          ,mCONTEXT,SEMANTIC_LABELS \
                                          ,mCONTEXT2,SEMANTIC_LABELS \
                                          ,mGLOBAL,SEMANTIC_LABELS\
                                          ,mCONLAB \
                                          ,mCONEDGES \
                                          ,mCONEDDIR \
                                          ,mHOG \
                                          ,mCONHOG).copyTo(des);
    
    
    //caffe
    
    if (mCAFFE == 1)
    {
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_SP_MASK).clone();
        Mat desCaf= _caffe->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);
        else
            des=desCaf.clone();
        imageSP.release();
        desCaf.release();
    }
    
    if (mCONCAFFE2 == 1)
    {
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_NEIG_MASK).clone();
        
        Mat desCaf= _caffe->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);
        else
            des=desCaf.clone();
        imageSP.release();
        desCaf.release();
    }
    //HYBRIDCNN
    if (mHYBRIDCNN == 1)
    {
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_SP_MASK).clone();
        
        Mat desCaf= _hybridcnn->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);
        else
            des=desCaf.clone();
        imageSP.release();
        desCaf.release();
    }
    if (mCONHYBRIDCNN == 1)
    {
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_NEIG_MASK).clone();
        
        Mat desCaf= _hybridcnn->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);
        else
            des=desCaf.clone();
        imageSP.release();
        desCaf.release();
    }

    return des;
}

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges,string dir_edgesDIR,string dir_segmen,
                  int nSAMPLES, string nameSVM, string dir_des, string filename, string dataset)
{
    
    clock_t start = clock();
    float timeSuperpixels=0;
    
    Mat trainingData,labels;
   
    ifstream infile(filename);
    
    if (! infile.good()) //si los datos de training no existen, los calculo
    {
        string model = "/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel";
        string proto = "/Users/acambra/Dropbox/test_caffe/deploy.prototxt" ;
        
        if (mCAFFE==1 || mCONCAFFE==1 || mCONCAFFE2 ==1)
            _caffe = new utilsCaffe(model,proto);
        
        if (mHYBRIDCNN==1 || mCONHYBRIDCNN==1)
           _hybridcnn = new utilsCaffe("/Users/acambra/TESIS/caffeModels/hybridCNN/hybridCNN_iter_700000.caffemodel",
                                        "/Users/acambra/TESIS/caffeModels/hybridCNN/hybridCNN_deploy.prototxt");
        
        
        //SVM
        int numDES = 0;
        
        if (mLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
        if (mCONLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
        if (mRGB == 1)      numDES += (3*NBINS_RGB);
        if (mPEAKS == 1)    numDES += NBINS_PEAKS;
        if (mEDGES == 1)    numDES += NBINS_EDGES;
        if (mCONEDGES == 1)    numDES += NBINS_EDGES;
        if (mEDDIR == 1)    numDES += NBINS_EDDIR;
        if (mCONEDDIR == 1)    numDES += NBINS_EDDIR;
        if (mHOG == 1)    numDES += NBINS_HOG;
        if (mCONHOG == 1)    numDES += NBINS_HOG;
        if (mCAFFE == 1)    numDES += NUMCAFFE;
        if (mCONCAFFE == 1)    numDES += NUMCAFFE;
        if (mCONCAFFE2 == 1)    numDES += NUMCAFFE;
        if (mSEMANTIC == 1) numDES += (SEMANTIC_LABELS);
        if (mCONTEXT == 1)  numDES += (SEMANTIC_LABELS);
        if (mCONTEXT2 == 1) numDES += (4*SEMANTIC_LABELS);
        if (mGLOBAL == 1)   numDES += (4*SEMANTIC_LABELS);
        if (mHYBRIDCNN == 1)   numDES += (NUMHYBRIDCNN);
        if (mCONHYBRIDCNN == 1)   numDES += (NUMHYBRIDCNN);
        
        int NEGIMG,NEGSAMPLES;
        float ACC_THRESHOLD;
        
        if (nSAMPLES == 333)
        {
            ACC_THRESHOLD = 1.0;
            NEGIMG = 2;
            NEGSAMPLES = (NEGIMG*800); //(nSAMPLES);//
        }
        else if (nSAMPLES == 933)
        {
            ACC_THRESHOLD = 0.75;
            NEGIMG = 4;
            NEGSAMPLES = (NEGIMG*800);
        }
        else if (nSAMPLES == 2022)
        {
            ACC_THRESHOLD = 0.5;
            NEGIMG = 8;
            NEGSAMPLES = (NEGIMG*800);
        }
        else if (nSAMPLES == 175) //svt1 acc 1
        {
            ACC_THRESHOLD = 1.0;
            NEGIMG = 5;
            NEGSAMPLES = (NEGIMG*100);
        }
        else if (nSAMPLES == 352) //svt1 acc 0.75
        {
            ACC_THRESHOLD = 0.75;
            NEGIMG = 10;
            NEGSAMPLES = (NEGIMG*100);
        }
        else if (nSAMPLES == 570) //svt1 acc 0.5
        {
            ACC_THRESHOLD = 0.5;
            NEGIMG = 17;
            NEGSAMPLES = (NEGIMG*100);
        }
        else if (nSAMPLES == 233) //doors pascal context acc 0.5
        {
            ACC_THRESHOLD = 0.5;
            NEGIMG = 21;
            NEGSAMPLES = (NEGIMG*32);
        }
        else
        {
            ACC_THRESHOLD = 0.5;
            NEGIMG = 1;//1;
            NEGSAMPLES = (nSAMPLES);//(NEGIMG*100);
        }
        
        //nSamples positives
        //aprox( 3 * nSamples) negatives
        // 800 es el NUMERO DE IMAGENES DE ENTRENAMIENTO!!!!!
        
        
        //Mat
        trainingData = Mat::zeros(nSAMPLES+NEGSAMPLES,numDES,CV_32FC1);
        //Mat
        labels = Mat::zeros(nSAMPLES+NEGSAMPLES, 1, CV_32FC1);//*/
        
        int pos=0;
        int neg=0;
        
        int numI=0;
        
        ////////////////////////
        //ADD POSITIVE SAMPLES: nSamples
        
        for (auto i = boost::filesystem::directory_iterator(dir_path); i != boost::filesystem::directory_iterator(); i++)
        {
            if (!is_directory(i->path()))
            {
                if ((pos + neg) >= (nSAMPLES + NEGSAMPLES))
                    break;//*/
                
                //string IMAGE
                string nameImage = dir_path + "/" + i->path().filename().string();
                string extension = i->path().filename().extension().string();
                
                if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
                {
                    printf("\n\n==================================================\n");
                    printf("**** Image (%d): %s (%d / 1000)\n",(pos + neg),i->path().filename().string().c_str(),++numI);
                    printf("==================================================\n");
                    
                   
                    //string GT
                    string imageGT = "";
                    
                    if (dir_pathGT != "")
                    {
                        imageGT = dir_pathGT + "/gt_" + i->path().filename().string();
                        size_t lastindex = imageGT.find_last_of(".");
                        imageGT = imageGT.substr(0, lastindex) + ".png";
                    }
                    
                    //string EDGES
                    string nameEdges = "";
                    if (mEDGES == 1 || mCONEDGES ==1)
                    {
                        nameEdges = dir_edges + "/" + i->path().filename().string();
                        nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                    }
                    //string EDGESDIR
                    string nameEdgesDIR = "";
                    if (mEDDIR == 1 || mCONEDDIR ==1)
                    {
                        nameEdgesDIR = dir_edgesDIR + "/" + i->path().filename().string();
                        nameEdgesDIR = nameEdgesDIR.substr(0, nameEdgesDIR.find_last_of(".")) + ".yml";
                        //edges too
                        nameEdges = dir_edges + "/" + i->path().filename().string();
                        nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                    }
                    
                    //string SEMANTIC
                    string nameSegmen = "";
                    if (mSEMANTIC == 1 || mCONTEXT == 1 || mCONTEXT2 == 1 || mGLOBAL == 1)
                    {
                        nameSegmen = dir_segmen + "/" + i->path().filename().string();
                        nameSegmen = nameSegmen.substr(0, nameSegmen.find_last_of(".")) + "_SegNet.png";
                    }
                    
                    
                    //SUPERPIXELS
                    SuperPixels *SPCTE;
                    SPCTE = svmSuperpixelsTEXT(nameImage,2,imageGT,nameSegmen);
                    
                   
                    //SAVE DESCRIPTORS

                    printf("---> Saving descriptors: %s\n",i->path().filename().string().c_str());
                    string dir_des = "/Users/acambra/TESIS/datasets/" + dataset +  "/train/BB_descriptors/";// + i->path().filename().string();
                    clock_t ini = clock();
                    saveSUPERPIXELSdescriptors(SPCTE,dir_des + i->path().filename().string(),nameEdges,nameEdgesDIR,nameSegmen);
                    float timeDescriptors = (float) (((double)(clock() - ini)) / CLOCKS_PER_SEC);
                     printf("**** Descriptors: %f seconds / %d numSuperpixels = %f s/sp\n ",timeDescriptors,SPCTE->maxID+1,timeDescriptors/(float)(SPCTE->maxID+1));
                    
                    Mat descriptors = Mat::zeros(SPCTE->maxID+1,numDES,CV_32FC1);
                    Mat accText = Mat::zeros(1, SPCTE->maxID+1, CV_32FC1);
                    
                    
                    calculateSUPERPIXELSdescriptors(SPCTE, nameEdges, nameEdgesDIR, nameSegmen, &descriptors, &accText,dir_des,i->path().filename().string());
                    
                    //sort random ID
                    std::vector<int> v;
                    for (int s=0; s < SPCTE->maxID+1; s++)
                        v.push_back(s);
                    
                    random_device rd;
                    mt19937 g(rd());
                    std::shuffle(v.begin(), v.end(), g);
                    
                    int sampleN = NEGIMG; //choose segun nSamples!!!!!
                    
                    
                    
                    for (int ind=0; ((ind < SPCTE->maxID+1)) ; ind++)
                    {
                        int id = v[ind];
                        float accT = accText.at<float>(id);//SPCTE->_arraySP[ind].accLabel(1);//
                       
                        int n = neg + pos;
                        
                        if ( accT >= ACC_THRESHOLD  && (pos < nSAMPLES))
                        {
                            descriptors.row(id).copyTo(trainingData.row(n));
                            labels.at<float>(n,0) = (float) LABEL_TEXT;
                            pos++;
                            printf("+ %d %f\n",pos,accT);
                            
                            //save positives!
                            /* string npos = "/Users/acambra/TESIS/datasets/svt1/train/pos05/" + i->path().filename().string() + to_string(id) + ".png";
                             //"/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Debug/ICDAR/train/pos1/" + i->path().filename().string() + to_string(id) + ".png";
                             imwrite(npos,SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));//*/
                            
                        }
                        else if (((1 - accT) == 1.0) && (sampleN > 0) && (neg < NEGSAMPLES))
                        {
                            descriptors.row(id).copyTo(trainingData.row(n));
                            labels.at<float>(n,0) = (float) LABEL_NOTEXT;//*/
                            neg++;
                            sampleN--;
                            printf("- %d %f\n",neg,accT);
                            
                            /* string nneg = "/Users/acambra/TESIS/datasets/svt1/train/neg/" + i->path().filename().string() + to_string(id) + ".png";
                             imwrite(nneg,SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));
                             //*/
                            
                        }
                    }//for SPTCTE//*/
                    
                    /*delete SPCTE;
                    descriptors.release();
                    accText.release();*/
                }
            }//if image
        }//for
        
        printf("TOTAL: pos : %d neg: %d",pos,neg);
        
        
        //SAVE TRAIN DESCRIPTORS
        //string filename = nameSVM + "_des.yaml";
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "trainingData" << trainingData;
        fs << "labels" << labels;
        fs.release();
        
        
    }
    else
    {
        printf("\n\t* Load training,labels files: %s\n\n", filename.c_str());
        //read  trainingData, labels from file
        FileStorage fs(filename, FileStorage::READ);
        fs["trainingData"] >> trainingData;
        fs["labels"] >> labels;
        fs.release();
    }
    
    
    //TRAIN SVM
    
    // Set up SVM's parameters
    //LINEAL
    CvSVMParams params;
    
    if (nameSVM.find("RBF2") != std::string::npos)
    {
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::RBF;
        params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, (int)1e8, (double)1e-6);
        params.gamma = 0.001;
    }
    else if ((nameSVM.find("RBF") != std::string::npos) || (nameSVM.find("AUTO") != std::string::npos))
    {
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::RBF;
        params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, (int)1e8, (double)1e-6);
        params.gamma = 0.1;
        /* params.degree = 0;
         params.gamma = 5.383;
         params.coef0 = 0;
         
         params.C = 2.67; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
         params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
         params.p = 0.0; // for CV_SVM_EPS_SVR*/
        
    }
    else
    {
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, (int)1e8, (double)1e-6);
    }
    
    clock_t startTrain = clock();
    
    
    // Train the SVM
    CvSVM SVM;
    
    if ((nameSVM.find("AUTO") != std::string::npos))
    {
        SVM.train_auto(trainingData, labels, Mat(), Mat(), params);
    }
    else
    {
        SVM.train(trainingData, labels, Mat(), Mat(), params);
    }
    SVM.save(nameSVM.c_str());
    SVM.clear();
    
    
    //}
    
    printf("\n===================================================================\n");
    printf("**** \tTRAIN %0.3f TOTAL %0.3f seconds\n",
           (float) (((double)(clock() - start)) / CLOCKS_PER_SEC),
           (float) (((double)(clock() - startTrain)) / CLOCKS_PER_SEC) );
    printf("===================================================================\n");
    printf("**** Superpixels: %f seconds\n ",timeSuperpixels);
   
    /* if (mLAB == 1)      printf("**** LAB: %f seconds\n ",timeLAB);
     if (mRGB == 1)      printf("**** RGB: %f seconds\n ",timeRGB);
     if (mEDGES == 1)    printf("**** EDGES: %f seconds\n ",timeEDGES);
     if (mEDDIR == 1)    printf("**** EDDIR: %f seconds\n ",timeEDGESDIR);
     if (mCAFFE == 1)    printf("**** CAFFE: %f seconds\n ",timeCAFFE);
     if (mSEMANTIC == 1) printf("**** SEMANTIC: %f seconds\n ",timeSEMANTIC);//*/
    
}

void calculateSUPERPIXELSdescriptors(SuperPixels *SPCTE, string nameEdges, string nameEdgesDIR, string nameSegmen, Mat *descriptors, Mat *accText, string path,string img)
{

    //READ????
    Mat desRGB,desLAB,desCONLAB,desEDGES,desEDDIR,desCONEDGES,desCONEDDIR,
        desHOG,desCONHOG,
        desCAFFE,desCONCAFFE,desCONCAFFE2,desSEMANTIC, desCONTEXT, desCONTEXT2, desGLOBAL,
        desHYBRIDCNN,desCONHYBRIDCNN;
    
    int num = SPCTE->maxID+1;
    
    // desSEMANTIC = Mat::zeros(num,2*(SEMANTIC_LABELS),CV_32FC1);*/
    bool readDescriptor = descriptorFileText(path,img, num,&desRGB,&desLAB,&desCONLAB,
                                             &desEDGES, &desCONEDGES,
                                             &desEDDIR, &desCONEDDIR,
                                             &desHOG,&desCONHOG,
                                             &desCAFFE,
                                             &desSEMANTIC,
                                             &desCONTEXT,&desCONTEXT2,&desGLOBAL,&desCONCAFFE,&desCONCAFFE2,
                                             &desHYBRIDCNN,&desCONHYBRIDCNN
                                             ,accText);
   
    //for each id in the image
    for (int id=0; (id < SPCTE->maxID+1) ; id++)
    {
        //all des together
        //Mat desID = descriptorText(SPCTE, id, nameEdges,nameEdgesDIR, nameSegmen);
         Mat desID;
        
        if (readDescriptor)
        {
            desID =  concatFileDescriptors(id,desRGB,desLAB,desCONLAB,
                                           desEDGES, desCONEDGES,
                                           desEDDIR, desCONEDDIR,
                                           desHOG,desCONHOG,
                                           desCAFFE, desSEMANTIC, desCONTEXT,desCONTEXT2,desGLOBAL,desCONCAFFE,desCONCAFFE2,
                                           desHYBRIDCNN,desCONHYBRIDCNN).clone();
        }
        else
            desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen).clone();
        
        desID.row(0).copyTo(descriptors->row(id));
        
        float acc = SPCTE->_arraySP[id].accLabel(1);
        accText->at<float>(0,id) = acc;
        
    }//for superpixels
}


void saveSUPERPIXELSdescriptors(SuperPixels *SPCTE, string img, string nameEdges, string nameEdgesDIR, string nameSegmen)
{
    //img = "/Users/acambra/TESIS/datasets/svt1/train/descriptors/" + img;
    //"/Users/acambra/TESIS/CODE/build/GibHub_test_superpixels/Debug/ICDAR/train/descriptors/" + img;
    
    string fRGB = img + "_RGB" + ".bin";
    string fLAB = img + "_LAB" + ".bin";
    string fCONLAB = img + "_CONLAB7" + ".bin";
    
    if (NBINS_L == 7 && NBINS_AB == 7)
        fLAB = img + "_LAB7" + ".bin";
    else if (NBINS_L == 9 && NBINS_AB == 9)
        fLAB = img + "_LAB9" + ".bin";
    
    string fEDGES = img + "_EDGES" + ".bin";
    string fEDDIR = img + "_EDDIR" + ".bin";
    string fCONEDGES = img + "_CONEDGES" + ".bin";
    string fCONEDDIR = img + "_CONEDDIR" + ".bin";
    
    string fHOG = img + "_HOG" + ".bin";
    string fCONHOG = img + "_CONHOG" + ".bin";
    
    string fCAFFE = img + "_CAFFE" + ".bin";
    string fCONCAFFE = img + "_CONCAFFE" + ".bin";
    string fCONCAFFE2 = img + "_CONCAFFE2" + ".bin";
    string fSEMANTIC = img + "_SEMANTIC" + ".bin";
    string fCONTEXT= img + "_CONTEXT" + ".bin";
    string fCONTEXT2= img + "_CONTEXT2" + ".bin";
    string fGLOBAL= img + "_GLOBAL" + ".bin";
    
    string fHYBRIDCNN= img + "_HYBRIDCNN" + ".bin";
    string fCONHYBRIDCNN= img + "_CONHYBRIDCNN" + ".bin";
    
    FILE *frgb,*flab,*fconlab,*fedges,*feddir,*fconedges,*fconeddir,
         *fhog,*fconhog,
         *fcaffe,*fsemantic,*fcontext,*fcontext2,*fglobal,*fconcaffe,*fconcaffe2
         ,*fhybridcnn,*fconhybridcnn;
    
    //EDGES
    Mat imgEdges;
    Mat imgEdgesDIR;

    
    //open all files
    ifstream infile(fLAB);
    if (! infile.good() && mLAB == 1)
    {
        flab = fopen(fLAB.c_str(), "wb");
    }
    
    ifstream infile11(fCONLAB);
    if (! infile11.good() && mCONLAB == 1)
    {
        fconlab = fopen(fCONLAB.c_str(), "wb");
    }
    
    //RGB: ID DES ACC
    ifstream infile1(fRGB);
    if (! infile1.good() && mRGB == 1)
    {
        frgb = fopen(fRGB.c_str(), "wb");
    }
    
    //EDGES: ID DES ACC
    ifstream infile2(fEDGES);
    if (! infile2.good() && mEDGES == 1)
    {
        fedges = fopen(fEDGES.c_str(), "wb");
        //EDGES
        imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
    }
    ifstream infile12(fCONEDGES);
    if (! infile12.good() && mCONEDGES == 1)
    {
        fconedges = fopen(fCONEDGES.c_str(), "wb");
        //EDGES
        imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
    }
    
    //EDDIR: ID DES ACC
    ifstream infile3(fEDDIR);
    if (! infile3.good() && mEDDIR == 1)
    {
        feddir= fopen(fEDDIR.c_str(), "wb");
        imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
        imgEdgesDIR = loadMatFromYML(nameEdgesDIR,"N");
    }
    ifstream infile13(fCONEDDIR);
    if (! infile13.good() && mCONEDDIR == 1)
    {
        fconeddir= fopen(fCONEDDIR.c_str(), "wb");
        imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
        imgEdgesDIR = loadMatFromYML(nameEdgesDIR,"N");
    }
    //HOG
    ifstream infile14(fHOG);
    if (! infile14.good() && mHOG == 1)
    {
        fhog= fopen(fHOG.c_str(), "wb");
    }
    ifstream infile15(fCONHOG);
    if (! infile15.good() && mCONHOG == 1)
    {
        fconhog= fopen(fCONHOG.c_str(), "wb");
    }
    
    
    //SEMANTIC: ID DES ACC
    ifstream infile4(fSEMANTIC);
    if (! infile4.good() && mSEMANTIC == 1)
    {
        fsemantic = fopen(fSEMANTIC.c_str(),"wb");
    }
    //CONTEXT: ID DES ACC
    ifstream infile5(fCONTEXT);
    if (! infile5.good() && mCONTEXT == 1)
    {
        fcontext = fopen(fCONTEXT.c_str(),"wb");
    }
    
    //CONTEXT2: ID DES ACC
    ifstream infile7(fCONTEXT2);
    if (! infile7.good() && mCONTEXT2 == 1)
    {
        fcontext2 = fopen(fCONTEXT2.c_str(),"wb");
    }
    
    
    //CONTEXT2: ID DES ACC
    ifstream infile8(fGLOBAL);
    if (! infile8.good() && mGLOBAL == 1)
    {
        fglobal = fopen(fGLOBAL.c_str(),"wb");
    }
    
    //CAFFE
    ifstream infile6(fCAFFE);
    if (! infile6.good() && mCAFFE == 1)
    {
        fcaffe = fopen(fCAFFE.c_str(), "wb");
    }
    ifstream infile9(fCONCAFFE);
    if (! infile9.good() && mCONCAFFE == 1)
    {
        fconcaffe = fopen(fCONCAFFE.c_str(), "wb");
    }
    
    ifstream infile10(fCONCAFFE2);
    if (! infile10.good() && mCONCAFFE2 == 1)
    {
        fconcaffe2 = fopen(fCONCAFFE2.c_str(), "wb");
    }
    
    //HyBRIFCNN
    ifstream infile16(fHYBRIDCNN);
    if (! infile16.good() && mHYBRIDCNN == 1)
    {
        fhybridcnn = fopen(fHYBRIDCNN.c_str(), "wb");
    }
    ifstream infile17(fCONHYBRIDCNN);
    if (! infile17.good() && mCONHYBRIDCNN == 1)
    {
        fconhybridcnn = fopen(fCONHYBRIDCNN.c_str(), "wb");
    }


    Mat desID=Mat();
    //for each id in the image
    for (int id=0; (id < SPCTE->maxID+1) ; id++)
    {
        
        float fid= (float) id;
        float acc = SPCTE->_arraySP[id].accLabel(1);
        
        if (! infile.good() && mLAB == 1)
        {
            //LAB: ID DES ACC
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,1,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS).clone();
            Descriptors::descriptors2file(flab, fid, desID, acc);
        }
        
        if ( !infile11.good() && mCONLAB == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,mLAB,NBINS_L,NBINS_AB \
                                                ,mRGB,NBINS_RGB\
                                                ,mPEAKS,NBINS_PEAKS \
                                                ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,mEDDIR,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,mSEMANTIC,SEMANTIC_LABELS \
                                                ,mCONTEXT,SEMANTIC_LABELS \
                                                ,mCONTEXT2,SEMANTIC_LABELS \
                                                ,mGLOBAL,SEMANTIC_LABELS\
                                                ,mCONLAB \
                                                ,mCONEDGES \
                                                ,mCONEDDIR);
            
            Descriptors::descriptors2file(fconlab, fid, desID, acc);
            
            
        }
        
        //RGB: ID DES ACC
        if (! infile1.good() && mRGB == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,1,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,0,SEMANTIC_LABELS\
                                                ,0,0,0).clone();
            Descriptors::descriptors2file(frgb, fid, desID, acc);
        }
        
        //EDGES: ID DES ACC
        if (! infile2.good() && mEDGES == 1)
        {
            imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,1,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,0,SEMANTIC_LABELS\
                                                ,0,0,0).clone();
            Descriptors::descriptors2file(fedges, fid, desID, acc);
        }
        if (! infile12.good() && mCONEDGES == 1)
        {
            imgEdges= imread(nameEdges,CV_LOAD_IMAGE_COLOR);
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,0,SEMANTIC_LABELS\
                                                ,0,1,0).clone();
            Descriptors::descriptors2file(fconedges, fid, desID, acc);
        }
        
        
        //EDDIR: ID DES ACC
        if (! infile3.good() && mEDDIR == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,1,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,1,SEMANTIC_LABELS\
                                                ,0,0,0).clone();
            Descriptors::descriptors2file(feddir, fid, desID, acc);
        }
        if (! infile13.good() && mCONEDDIR == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,0,SEMANTIC_LABELS\
                                                ,0,0,1).clone();
            Descriptors::descriptors2file(fconeddir, fid, desID, acc);
        }
        
        //HOG
        if (! infile14.good() && mHOG == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,mLAB,NBINS_L,NBINS_AB \
                                                ,mRGB,NBINS_RGB\
                                                ,mPEAKS,NBINS_PEAKS \
                                                ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,mEDDIR,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,mSEMANTIC,SEMANTIC_LABELS \
                                                ,mCONTEXT,SEMANTIC_LABELS \
                                                ,mCONTEXT2,SEMANTIC_LABELS \
                                                ,mGLOBAL,SEMANTIC_LABELS\
                                                ,mCONLAB \
                                                ,mCONEDGES \
                                                ,mCONEDDIR \
                                                ,mHOG \
                                                ,mCONHOG).clone();
            
            Descriptors::descriptors2file(fhog, fid, desID, acc);
        }
        if (! infile15.good() && mCONHOG == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,mLAB,NBINS_L,NBINS_AB \
                                                ,mRGB,NBINS_RGB\
                                                ,mPEAKS,NBINS_PEAKS \
                                                ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,mEDDIR,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,mSEMANTIC,SEMANTIC_LABELS \
                                                ,mCONTEXT,SEMANTIC_LABELS \
                                                ,mCONTEXT2,SEMANTIC_LABELS \
                                                ,mGLOBAL,SEMANTIC_LABELS\
                                                ,mCONLAB \
                                                ,mCONEDGES \
                                                ,mCONEDDIR \
                                                ,mHOG \
                                                ,mCONHOG).clone();
            Descriptors::descriptors2file(fconhog, fid, desID, acc);
        }
        
        
        
        //SEMANTIC: ID DES ACC
        if (! infile4.good() && mSEMANTIC == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,1,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,0,SEMANTIC_LABELS\
                                                ,0,0,0).clone();
            Descriptors::descriptors2file(fsemantic, fid, desID, acc);
            
            /* for (int i=0; i<desID.cols; i++)
             printf("id %d escrito: %f \n",id,desID.at<float>(i));//*/
        }
        
        //CONTEXT: ID DES ACC
        if (! infile5.good() && mCONTEXT == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,1,SEMANTIC_LABELS).clone();
            Descriptors::descriptors2file(fcontext, fid, desID, acc);
        }
        
        if (! infile7.good() && mCONTEXT2 == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,1,SEMANTIC_LABELS).clone();
            Descriptors::descriptors2file(fcontext2, fid, desID, acc);
        }
        
        if (! infile8.good() && mGLOBAL == 1)
        {
            desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS\
                                                ,0,SEMANTIC_LABELS \
                                                ,1,SEMANTIC_LABELS).clone();
            Descriptors::descriptors2file(fglobal, fid, desID, acc);
        }
        
        //CAFFE
        if (! infile6.good() && mCAFFE == 1)
        {
            
            Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_SP_MASK).clone();
            Mat desCaf= _caffe->features(imageSP, "fc7").clone();
            normalize(desCaf, desCaf);
            
            Descriptors::descriptors2file(fcaffe, fid, desCaf, acc);
        }
        if (! infile9.good() && mCONCAFFE == 1)
        {
            
            Mat imageSP = SPCTE->getImage().clone();
            Mat mask = SPCTE->getMaskNeigbours(id);
            
            Mat nonZeroCoordinates;
            findNonZero( mask, nonZeroCoordinates);
            
            double minX=imageSP.cols, minY=imageSP.rows, maxX=0.0,maxY=0.0;
            
            for (int i = 0; i < nonZeroCoordinates.total(); i++ )
            {
                if (nonZeroCoordinates.at<Point>(i).x <= minX) minX = nonZeroCoordinates.at<Point>(i).x;
                else if (nonZeroCoordinates.at<Point>(i).x >= maxX) maxX =  nonZeroCoordinates.at<Point>(i).x;
                
                if (nonZeroCoordinates.at<Point>(i).y <= minY) minY = nonZeroCoordinates.at<Point>(i).y;
                else if (nonZeroCoordinates.at<Point>(i).y >= maxY) maxY =  nonZeroCoordinates.at<Point>(i).y;
            }
            
            Mat maskID = SPCTE->_arraySP[id].getMask().clone();
            bitwise_not(maskID, maskID);
            cvtColor(maskID,maskID,CV_GRAY2BGR);
            bitwise_and(imageSP,maskID, imageSP);
            
            Mat roi = imageSP(Rect(minX,minY,maxX-minX, maxY-minY)).clone();
            
            Mat desCaf= _caffe->features(roi, "fc7").clone();
            normalize(desCaf, desCaf);
            
            Descriptors::descriptors2file(fconcaffe, fid, desCaf, acc);
        }
        if (! infile10.good() && mCONCAFFE2 == 1)
        {
            Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_NEIG_MASK).clone();
            
            Mat desCaf= _caffe->features(imageSP, "fc7").clone();
            normalize(desCaf, desCaf);
            
            Descriptors::descriptors2file(fconcaffe2, fid, desCaf, acc);
        }
        
        //HYBRID
        if (! infile16.good() && mHYBRIDCNN == 1)
        {
            
            Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_SP_MASK).clone();
            Mat desCaf= _hybridcnn->features(imageSP, "fc7").clone();
            normalize(desCaf, desCaf);
            
            Descriptors::descriptors2file(fhybridcnn, fid, desCaf, acc);
        }
        if (! infile17.good() && mCONHYBRIDCNN == 1)
        {
            Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1,BB_NEIG_MASK).clone();
            
            Mat desCaf= _hybridcnn->features(imageSP, "fc7").clone();
            normalize(desCaf, desCaf);
            
            Descriptors::descriptors2file(fconhybridcnn, fid, desCaf, acc);
        }

    }//for superpixels
    
    printf("\tSaved descriptors: %s\n",img.c_str());
    
    
    if (! infile.good() && mLAB == 1) fclose(flab);
    
    if (! infile1.good() && mRGB == 1) fclose(frgb);
    
    if (! infile2.good() && mEDGES == 1) fclose(fedges);
    if (! infile12.good() && mCONEDGES == 1) fclose(fconedges);
    
    if (! infile3.good() && mEDDIR == 1) fclose(feddir);
    if (! infile13.good() && mCONEDDIR == 1) fclose(fconeddir);
    
    if (! infile14.good() && mHOG == 1) fclose(fhog);
    if (! infile15.good() && mCONHOG == 1) fclose(fconhog);
    
    if (! infile4.good() && mSEMANTIC == 1) fclose(fsemantic);
    
    if (! infile5.good() && mCONTEXT == 1) fclose(fcontext);
    
    if (! infile7.good() && mCONTEXT2 == 1) fclose(fcontext2);
    
    if (! infile8.good() && mGLOBAL == 1) fclose(fglobal);
    
    if (! infile6.good() && mCAFFE == 1)  fclose(fcaffe);
    
    if (! infile9.good() && mCONCAFFE == 1)  fclose(fconcaffe);
    
    if (! infile10.good() && mCONCAFFE2 == 1)  fclose(fconcaffe2);
    
    if (! infile11.good() && mCONLAB == 1)  fclose(fconlab);
    
    if (! infile16.good() && mHYBRIDCNN == 1)  fclose(fhybridcnn);
    if (! infile17.good() && mCONHYBRIDCNN == 1)  fclose(fconhybridcnn);

    
}
