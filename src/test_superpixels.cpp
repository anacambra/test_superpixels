
//
//  main.cpp
//  test_superpixels
//
//  Created by Ana Cambra on 26/11/14.
//  Copyright (c) 2014 Ana Cambra. All rights reserved.
//

#include <iostream>

#include "superpixels.cpp"
//#include <time.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <random>

using namespace cv;

using namespace boost;

//using namespace boost::filesystem;
//#include <caffe/caffe.hpp>
//using namespace caffe;


#define LABEL_TEXT 1
#define LABEL_NOTEXT -1

//#define DEBUG_SP 1

int mLAB   = 0; int NBINS_L = 50; int NBINS_AB=128;
int mRGB   = 0; int NBINS_RGB   = 256;
int mPEAKS = 0; int NBINS_PEAKS = 128;

int mLINES = 0; int NBINS_LINES = 128;

int mEDGES = 0; int NBINS_EDGES; int modeEDGES = 1;
int mEDDIR = 0; int NBINS_EDDIR = 8;

int mCAFFE = 0; string CAFFE_LAYER = "fc7"; int NUMCAFFE = 4096;

int mSEMANTIC = 0; int SEMANTIC_LABELS = 12; //SEGNET
int SEMANTIC2_LABELS = 23;
int SEMANTIC3_LABELS = 22;

int mCONTEXT = 0; // semantic_neigbourg

bool DEBUG = false;

//caffe
utilsCaffe *_caffe;

/*int nSAMPLES = 50;
 string nameSVM = "svm_LAB_PEAKS_" + to_string(nSAMPLES) + "_NOTZERO.xml";*/

Mat concatFileDescriptors(int id,Mat desRGB,Mat desLAB,Mat desEDGES,Mat desEDDIR, Mat desCAFFE, Mat desSEMANTIC)
{
    Mat des;
    
    if (mLAB != 0)
    {
        des = desLAB.row(id);
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
    if (mEDDIR != 0)
    {
        if (des.rows != 0)
            hconcat(desEDDIR.row(id), des,des);
        else
            des=desEDDIR.row(id);
    }
    
    
    if (mCAFFE != 0)
    {
        if (des.rows != 0)
            hconcat(desCAFFE.row(id), des,des);
        else
            des=desCAFFE.row(id);
    }
    
    if (mSEMANTIC != 0)
    {
        if (des.rows != 0)
            hconcat(desSEMANTIC.row(id), des,des);
        else
            des=desSEMANTIC.row(id);
        
    }
    return des;
}



//train SVM
void descriptors2file(FILE *fout, float fid, Mat desID, float acc);

bool file2descriptors(string file, int size, Mat *descriptors, Mat *accText);
bool descriptorFileText(string path,string img,int numID,Mat *desRGB,Mat *desLAB,Mat *desEDGES,Mat *desEDDIR, Mat *desCAFFE, Mat *desSEMANTIC,Mat *accText);
void saveSUPERPIXELSdescriptors(SuperPixels *SPCTE, string img, string nameEdges, string nameEdgesDIR, string nameSegmen);
void calculateSUPERPIXELSdescriptors(SuperPixels *SPCTE, string nameEdges, string nameEdgesDIR, string nameSegmen, Mat *descriptors, Mat *accText, string path, string img);

Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges = "", string nameEdgesDIR="", string nameSegmen = "");

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges, string dir_edgesDIR, string dir_segmen , int nSAMPLES, string nameSVM, string dir_des);


//initialize structure of Superpixels
SuperPixels* svmSuperpixelsTEXT( string nameImage, int numLabels = 2, string imageGT = "", string nameSegmen = "")
{
    
    SuperPixels *SPCTE = new SuperPixels(nameImage);
    //boundaries between SP
    SPCTE->calculateBoundariesSuperpixels();
    //init superpixels
    SPCTE->initializeSuperpixels();
    //TEXT LABELS
    SPCTE->setNUMLABELS(2);
    
    if (!imageGT.empty())
        SPCTE->initializeLabeling(imageGT, MODE_LABEL_NOTZERO);//MODE_LABEL_MEDIAN);//
    
    // if (mCAFFE == 1) SPCTE->initCaffe("/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel","/Users/acambra/Dropbox/test_caffe/deploy.prototxt");
    
    SPCTE->setNUMLABELS(numLabels);
    
    if (!nameSegmen.empty())
        SPCTE->initializeSegmentation(nameSegmen,SEMANTIC_LABELS,MODE_LABEL_NOTZERO,12);
    
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

void cmpImage(string image, string gt, float* precision, float *recall)
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
    //imshow("gt",imgGT);
    
    Mat imgTP,imgFP,imgFN;
    
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
    
    //printf("P:%f R:%f \n",(*precision),(*recall));

}

void parseTXT(string name, string out)
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
    fprintf(fout,"              OPTIONS                      time             percentage          precision             recall             image\n");
    fprintf(fout,"---------------------------------------------------------------------------------------------------------------------------------\n");
    
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
            pos1=line3.find("img_");
            pos2=line3.find(".");
            string file = line3.substr(pos1,(pos2-pos1));
            
            float precision;
            float recall;
            string resul = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/" + string(imgFile);
            string gt = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/ICDAR/train/gt/gt_" + file + ".png";
            cmpImage(resul, gt, &precision, &recall);
            
            
            fprintf(fout,"%-40s%-20f%-20f%-20f%-20f%-20s\n",line2,time,percentage,precision,recall,file.c_str());//getchar();
            
            // getline(inFile,line2);
            //
            //break;
        }
    }
    fclose(fout);
}


int main(int argc, const char * argv[]) {
    
   /* matGTfromFileBoundingBox("/Users/acambra/Dropbox/dataset/ICDAR/ch4_training_localization_transcription_gt/us-ascii/" //gt_img_1.txt"//);
    ,
                            "/Users/acambra/Dropbox/dataset/ICDAR/ch4_images_gt");
    return 1;//*/
   
    //tp pn nn np
    
    /*cmpImage("/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/ICDAR/test/out/img_1000.jpg_CAFFE_500.png",
             "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/ICDAR/test/gt/gt_img_1000.png");//*/
    
    
    
    /*parseTXT("/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/test2_TRAIN_500.txt",
             "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/resul2_test_TRAIN_500.txt");
    return 1;//*/
    
    //parse argv
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
    ("svmTestDIR", boost::program_options::value<std::string>(), "dir images test")
    ("nSamples", boost::program_options::value<int>()->required(), "test svm")
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
    
    
    if (parameters["svmOptions"].as<std::string>().find("LAB") != std::string::npos)
    {
        mLAB = 1;
    }
    
    if (parameters["svmOptions"].as<std::string>().find("RGB") != std::string::npos)
        mRGB = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("PEAKS") != std::string::npos)
        mPEAKS = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("EDDIR") != std::string::npos)
        mEDDIR = 1;
    
    if (parameters["svmOptions"].as<std::string>().find("EDGES") != std::string::npos)
    {
        mEDGES = 1;
        
        if (modeEDGES == 0) //hist
            NBINS_EDGES = 100;
        else if (modeEDGES == 1) //moments
            NBINS_EDGES = 4;
        else if (modeEDGES == 2)
            NBINS_EDGES = 100 + 4;
        
    }
    
    
    if (parameters["svmOptions"].as<std::string>().find("SEMANTIC2") != std::string::npos)
    {
        mSEMANTIC = 1; SEMANTIC_LABELS = SEMANTIC2_LABELS;
    }
    else  if (parameters["svmOptions"].as<std::string>().find("SEMANTIC3") != std::string::npos)
    {
        mSEMANTIC = 1; SEMANTIC_LABELS = SEMANTIC3_LABELS;
    } else if (parameters["svmOptions"].as<std::string>().find("SEMANTIC") != std::string::npos)
    {
        mSEMANTIC = 1;
    }
    
    string svmType="";
     if (parameters["svmTrain"].as<std::string>().length() > 0)
         svmType = parameters["svmTrain"].as<std::string>();
     if (parameters["svmTest"].as<std::string>().length() > 0)
            svmType= parameters["svmTest"].as<std::string>();
    
    if (svmType.length()>0)
        svmType =  "_" + svmType;
    
    nameSVM = string("svm_") + parameters["svmOptions"].as<std::string>() + string("_") + to_string(nSAMPLES) + svmType + string("_NOTZERO.xml");
    string nameSVM2 = string("train/") + nameSVM;
    
    // printf("SVM %s. %s.\n", nameSVM.c_str(),nameSVM2.c_str() ); getchar();
    
    
    
    ////////////////////////
    //SVM train o load file
     ////////////////////////
    
    if (parameters.count("svmTrain")) {
        
        string svmType= parameters["svmTrain"].as<std::string>();
        printf("TYPE: %s\n",svmType.c_str());//getchar();
        
        printf("==================================================\n");
        printf("**** \tTRAIN SVM_%s numSamples %d\n",parameters["svmOptions"].as<std::string>().c_str(),nSAMPLES);
        printf("==================================================\n");
        
        ifstream infile(nameSVM2);
        if (! infile.good()) {
            
            if (parameters["svmOptions"].as<std::string>().find("CAFFE") != std::string::npos)
            {
                mCAFFE = 1;
                //initCaffe
                string model = "/Users/acambra/Dropbox/test_caffe/bvlc_reference_caffenet.caffemodel";
                string proto = "/Users/acambra/Dropbox/test_caffe/deploy.prototxt" ;
                //printf("Model CAFFE: %s PROTO: %s\n",model.c_str(),proto.c_str());getchar();
                
                _caffe = new utilsCaffe(model,proto);
            }
            
            string path = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/";
            
            trainSVMText(/*path + "ICDAR/train/images",
                         path + "ICDAR/train/gt",
                         path + "ICDAR/train/edges",
                         path + "ICDAR/train/normals",
                        /* path + "ICDAR/train/semantic",*/
                        /*"/Users/acambra/SegNet/ICDAR-SegNet",
                         nSAMPLES,
                         nameSVM2,
                         path + "train/ICDAR/descriptors");*/
            path + "ICDAR/test/images",
            "",
            path + "ICDAR/test/edges",
            path + "ICDAR/test/normals",
            /* path + "ICDAR/train/semantic",*/
            "/Users/acambra/SegNet/ICDAR-SegNet",
            nSAMPLES,
            nameSVM2,
            path + "train/ICDAR/descriptors");
            
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
    
    //if (parameters.find("svmTest") != parameters.end()) {
    if (parameters.count("svmTest")) {
      
        CvSVM SVM;
        SVM.load(nameSVM2.c_str());
        
        boost::filesystem::path inputImage(parameters["image"].as<std::string>());
        
        ////////////////////////
        //SUPERPIXELS
        
        
        string nameImage = inputImage.string();
        string nameGT="";
       
        string nameSegmen ="";
        if (mSEMANTIC == 1)
            nameSegmen = parameters["semantic"].as<std::string>();
        
        SuperPixels *SPCTE;
        
        SPCTE= svmSuperpixelsTEXT(nameImage,2,nameGT,nameSegmen);
        
        Mat imgSP = SPCTE->getImage();//Superpixels().clone();
        /*imshow("superpixels",SPCTE->getImageSuperpixels());
         waitKey(0);//*/
        
        //check neigbour
        // Mat image
        /*for (int id=0; id < SPCTE->maxID+1; id++)
         {
         imshow("superpixels",SPCTE->paintSuperpixel(SPCTE->getImageSuperpixels(),id));
         waitKey(25);
         }*/
        
        
        char k=-1;
        
        string nameWindow = "Superpixel ";
        
        //test
        
        if (parameters.find("labeled") != parameters.end())
            nameGT = parameters["labeled"].as<std::string>();
        
        int numText=0;
        int numNOText=0;
        clock_t start = clock();
        
        if (parameters["svmOptions"].as<std::string>().find("CAFFE") != std::string::npos)
        {
            mCAFFE = 1;
        }
        
        //READ????
        Mat desRGB,desLAB,desEDGES,desEDDIR,desCAFFE,desSEMANTIC,accText;
        int num = SPCTE->maxID+1;
        string file = nameImage.substr((int)nameImage.find_last_of("/")+1);
        
        string dir_des=nameImage.substr(0,(int)nameImage.find_last_of("/"));
        dir_des=dir_des.substr(0,(int)dir_des.find_last_of("/"))+ "/descriptors/";
        
       // desSEMANTIC = Mat::zeros(num,2*(SEMANTIC_LABELS),CV_32FC1);*/
        bool readDescriptor = descriptorFileText(dir_des,file, num,&desRGB,&desLAB,&desEDGES,&desEDDIR,&desCAFFE,&desSEMANTIC,&accText);
        
        if (mCAFFE == 1 && !readDescriptor)
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
            
            string nameEdges="";
            if (parameters.find("edges") != parameters.end())
                nameEdges = parameters["edges"].as<std::string>();
            
            string nameEdgesDIR="";
            if (parameters.find("eddir") != parameters.end())
                nameEdgesDIR = parameters["eddir"].as<std::string>();
            
            string nameSegmen="";
            if (parameters.find("semantic") != parameters.end())
                nameSegmen = parameters["semantic"].as<std::string>();
            
            //DESCRIPTOR SUPERPIXEL: calculate or read
            Mat desID;//,desF;
            
            
            //Mat desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen);
            
            if (readDescriptor)
                desID =  concatFileDescriptors(id,desRGB,desLAB,desEDGES, desEDDIR,  desCAFFE,  desSEMANTIC);//desSEMANTIC.row(id);

            else
                desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen);

            
            /*for (int i=0; i<desID.cols; i++)
                printf("calculado: %f leido: %f  \n",desID.at<float>(0,i),desF.at<float>(0,i));//*/
            //
            
            //evaluate SVM
            //float threshold = 0.5;
            if (SVM.get_var_count() != 0)
            {
                float response = SVM.predict(desID);//,true);
                printf("RESPONSE SVM id: %d  %f\n",id,response);
                //paint image with SVM response
                if (response >= (LABEL_TEXT - 0.5))
                {
                    imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();
                    
                    numText += SPCTE->numPixels(id);
                    if (DEBUG) printf("\t*** SVM response id: %d  TEXT (%f)\n",id,response);
                }
                else
                {
                    if (DEBUG) printf("\t*** SVM response id: %d  NO TEXT (%f)\n",id,response);
                    numNOText += SPCTE->numPixels(id);
                }
            }
            else
            {
                if (DEBUG) printf("** ERROR: SVM not found: %s \n",nameSVM.c_str());
            }
            
            if (DEBUG == 1) {
                
                destroyWindow(nameWindow.c_str());
                nameWindow = "Superpixel " + to_string(id);
                imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));
                
                //
                
                /* imshow(nameWindow.c_str(),SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels().clone(),id,3));
                 Mat img_sp;
                 Mat imgEdges = imread(parameters["edges"].as<std::string>(),CV_LOAD_IMAGE_COLOR);
                 //imshow("img EDGES ",imgEdges);
                 
                 Mat desID = descriptorText(SPCTE,id);//= SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                 ,mLAB,NBINS_L,NBINS_AB \
                 ,mRGB,NBINS_RGB\
                 ,mPEAKS,NBINS_PEAKS \
                 ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                 ,mCAFFE,CAFFE_LAYER,NUMCAFFE\
                 ,mSEMANTIC,SEMANTIC_LABELS).clone();*/
                
                
                
                //evaluate
                /* float response = SVM.predict(desID);//,true);
                 printf("RESPONSE SVM id: %d  %f\n",id,response);
                 
                 //paint image with SVM response
                 if (response >= LABEL_TEXT)
                 imgSP = SPCTE->paintSuperpixel(imgSP, id).clone();//*/
                
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
        if (mLAB == 1)      printf("**** LAB: %f seconds\n ",SPCTE->timeLAB);
        if (mRGB == 1)      printf("**** RGB: %f seconds\n ",SPCTE->timeRGB);
        if (mEDGES == 1)    printf("**** EDGES: %f seconds\n ",SPCTE->timeEDGES);
        if (mEDDIR == 1)    printf("**** EDDIR: %f seconds\n ",SPCTE->timeEDGESDIR);
        if (mCAFFE == 1)    printf("**** CAFFE: %f seconds\n ",SPCTE->timeCAFFE);
        if (mSEMANTIC == 1) printf("**** SEMANTIC: %f seconds\n ",SPCTE->timeSEMANTIC);//*/
        
        printf("-------------------------------------------------------------------\n");
        printf("**** \tTEXT CANDIDATES: %d (%0.2f %%)\n", numText,
               100.0 * (float)numText / (float)(SPCTE->getImage().rows*SPCTE->getImage().cols));
        
        if (DEBUG == 1){
            imshow(nameSVM,imgSP);waitKey(0);}
        else{
            
            size_t found1 = nameImage.find_last_of("/");
            string name = "out/"+ nameImage.substr(found1+1) + "_" + parameters["svmOptions"].as<std::string>()+ "_" + to_string(nSAMPLES)+ svmType +".png";
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
bool descriptorFileText(string path, string img,int numID, Mat *desRGB,Mat *desLAB,Mat *desEDGES,Mat *desEDDIR, Mat *desCAFFE, Mat *desSEMANTIC,Mat *accText)
{
    //check Files
    img = path + img;
    
    string fRGB = img + "_RGB" + ".bin";
    string fLAB = img + "_LAB" + ".bin";
    string fEDGES = img + "_EDGES" + ".bin";
    string fEDDIR = img + "_EDDIR" + ".bin";
    string fCAFFE = img + "_CAFFE" + ".bin";
    string fSEMANTIC = img + "_SEMANTIC" + ".bin";
    
   // FILE* frgb,*flab,*fedges,*feddir,*fcaffe,*fsemantic;
    
    int numDES = 0;
    
    //Mat desRGB,desLAB,desEDGES,desEDDIR, desCAFFE;//, desSEMANTIC;
    (* accText) = Mat::zeros(1, numID, CV_32FC1);
    
    bool ok = true;

    
    if (mRGB == 1 && ok)
    {
        numDES = (3*NBINS_RGB);
        (*desRGB) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fRGB,numDES,desRGB, accText);
    }
    
    if (mLAB == 1 && ok)
    {
        numDES =  NBINS_L + (2*NBINS_AB);
        (*desLAB) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fLAB,numDES,desLAB, accText);
    }
    
    if (mEDGES == 1 && ok)
    {
        numDES = NBINS_EDGES;
        (*desEDGES) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fEDGES,numDES,desEDGES, accText);
    }
    
    if (mEDDIR == 1 && ok)
    {
        numDES = NBINS_EDDIR;
        (*desEDDIR) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fEDDIR,numDES,desEDDIR, accText);
    }
    
    if (mCAFFE == 1 && ok)
    {
        numDES = NUMCAFFE;
        (*desCAFFE) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fCAFFE,numDES,desCAFFE, accText);
    }
    
    if (mSEMANTIC == 1)
    {
        numDES = 2*(SEMANTIC_LABELS);
        (*desSEMANTIC) = Mat::zeros(numID,numDES,CV_32FC1);
        ok = ok && file2descriptors(fSEMANTIC,numDES,desSEMANTIC,accText);
    }
    
    return ok;//desSEMANTIC->row(id).clone();
}


Mat descriptorText(SuperPixels *SPCTE, int id, string nameEdges,string nameEdgesDIR, string nameSegmen)
{
    //show info
    
    if (DEBUG)
    {
        string info= "";
        if (mLAB != 0)
            info += "LAB ";
        
        if (mRGB != 0)
            info += "RGB ";
        
        if (mPEAKS != 0)
            info += "PEAKS ";
        
        if (mEDGES != 0)
            info += "EDGES  ";
        if (mEDDIR != 0)
            info += "EDDIR  ";
        
        if (mCAFFE != 0)
            info += "CAFFE ";
        
        if (mSEMANTIC != 0)
            info += "SEMANTIC";
        printf("descriptors: %s id %d/%d \n",info.c_str(),id,SPCTE->maxID);
    }
    
    //EDGES
    Mat imgEdges = Mat();
    if (mEDGES == 1)
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
    if (mEDDIR == 1)
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
    
    //SEMANTIC
    Mat imgSegmen = Mat();
    if (mSEMANTIC == 1)
    {
        imgSegmen = imread(nameSegmen,CV_LOAD_IMAGE_GRAYSCALE);//COLOR
        if (imgSegmen.data == NULL)
        {
            printf("ERROR: No SEMANTIC SEGMENTATION image found\n");
        }
    }
    
    
    Mat des = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                          ,mLAB,NBINS_L,NBINS_AB \
                                          ,mRGB,NBINS_RGB\
                                          ,mPEAKS,NBINS_PEAKS \
                                          ,mEDGES,NBINS_EDGES,modeEDGES,imgEdges\
                                          ,mEDDIR,NBINS_EDDIR,imgEdgesDIR\
                                          ,0,CAFFE_LAYER,NUMCAFFE\
                                          ,mSEMANTIC,SEMANTIC_LABELS,imgSegmen).clone();
    //caffe
    
    if (mCAFFE == 1)
    {
        clock_t start = clock();
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1).clone();
        
        Mat desCaf= _caffe->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        /* double min, max;
         minMaxLoc(desCaf, &min, &max);
         Scalar     mean, stddev;
         meanStdDev ( desCaf, mean, stddev );
         
         printf("Caffe des %d values[%f,%f] mean %f stdDev %f \n",i, min, max,mean.val[0],stddev.val[0]);*/
        
        if (des.rows != 0)
            hconcat(desCaf,des,des);//_arraySP[i].descriptorsCAFFE(imageSP,CAFFE_LAYER,NUMCAFFE), des,des);
        else
            des=desCaf.clone();//_arraySP[i].descriptorsCAFFE(imageSP,CAFFE_LAYER,NUMCAFFE).clone();
        
        //printf("Descriptors Caffe %d\n",i);
        SPCTE->timeCAFFE += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
        
        //imageSP.release();
        //desCaf.release();*/
        
    }
    
    return des;
    
}

void trainSVMText(string dir_path,string dir_pathGT, string dir_edges,string dir_edgesDIR,string dir_segmen, int nSAMPLES, string nameSVM, string dir_des)
{
    //time
    float timeSuperpixels = 0.0;
    float timeLAB = 0.0;
    float timeRGB = 0.0;
    float timeEDGES = 0.0;
    float timeEDGESDIR = 0.0;
    float timeCAFFE = 0.0;
    float timeSEMANTIC = 0.0;
    
    //SVM
    int numDES = 0;
    
    if (mLAB == 1)      numDES += NBINS_L + (2*NBINS_AB);
    if (mRGB == 1)      numDES += (3*NBINS_RGB);
    if (mPEAKS == 1)    numDES += NBINS_PEAKS;
    if (mEDGES == 1)    numDES += NBINS_EDGES;
    if (mEDDIR == 1)    numDES += NBINS_EDDIR;
    if (mCAFFE == 1)    numDES += NUMCAFFE;
    if (mSEMANTIC == 1) numDES += 2*(SEMANTIC_LABELS);
    
    
    Mat trainingData = Mat::zeros(nSAMPLES*2,numDES,CV_32FC1);
    Mat labels = Mat::zeros(nSAMPLES*2, 1, CV_32FC1);
    
    int pos=0;
    int neg=0;
    
    int numI=0;
    
    clock_t start = clock();
    
    for (auto i = boost::filesystem::directory_iterator(dir_path); i != boost::filesystem::directory_iterator() && (pos+neg < nSAMPLES*2); i++)
    {
        if (!is_directory(i->path()))
        {
            if ((pos+neg >= nSAMPLES*2))
                break;
            
            //string IMAGE
            string nameImage = dir_path + "/" + i->path().filename().string();
            string extension = i->path().filename().extension().string();
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                printf("\n\n==================================================\n");
                printf("**** Image (%d): %s (%d / 1000)\n",(pos + neg),i->path().filename().string().c_str(),++numI);
                printf("==================================================\n");
                
                //if (numI < 72) continue;
                //string GT
                string imageGT = "";
                
                if (dir_pathGT != "")
                {
                    dir_pathGT + "/gt_" + i->path().filename().string();
                    size_t lastindex = imageGT.find_last_of(".");
                    imageGT = imageGT.substr(0, lastindex) + ".png";
                }
                
                //string EDGES
                string nameEdges = "";
                if (mEDGES == 1)
                {
                    nameEdges = dir_edges + "/" + i->path().filename().string();
                    nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                }
                //string EDGESDIR
                string nameEdgesDIR = "";
                if (mEDDIR == 1)
                {
                    nameEdgesDIR = dir_edgesDIR + "/" + i->path().filename().string();
                    nameEdgesDIR = nameEdgesDIR.substr(0, nameEdgesDIR.find_last_of(".")) + ".yml";
                    //edges too
                    nameEdges = dir_edges + "/" + i->path().filename().string();
                    nameEdges = nameEdges.substr(0, nameEdges.find_last_of(".")) + ".png";
                }
                
                //string SEMANTIC
                string nameSegmen = "";
                if (mSEMANTIC == 1)
                {
                    nameSegmen = dir_segmen + "/" + i->path().filename().string();
                    nameSegmen = nameSegmen.substr(0, nameSegmen.find_last_of(".")) + "_SegNet.png";
                }
                
                
                //SUPERPIXELS
                SuperPixels *SPCTE;
                SPCTE = svmSuperpixelsTEXT(nameImage,2,imageGT,nameSegmen);
                

                //SAVE DESCRIPTORS
                if (nameSVM.find("save") != std::string::npos)
                {
                    printf("---> Saving descriptors: %s\n",i->path().filename().string().c_str());
                    saveSUPERPIXELSdescriptors(SPCTE,i->path().filename().string(),nameEdges,nameEdgesDIR,nameSegmen);
                    
                }
                else
                {
                    Mat descriptors = Mat::zeros(SPCTE->maxID+1,numDES,CV_32FC1);
                    Mat accText = Mat::zeros(1, SPCTE->maxID+1, CV_32FC1);
                    
                    calculateSUPERPIXELSdescriptors(SPCTE, nameEdges, nameEdgesDIR, nameSegmen, &descriptors, &accText,dir_des,i->path().filename().string());
                    
                    //choose random ID
                    
                    std::vector<int> v;
                    for (int i=0; i < SPCTE->maxID+1; i++)
                        v.push_back(i);
                    
                    random_device rd;
                    mt19937 g(rd());
                    
                    std::shuffle(v.begin(), v.end(), g);
                    
                    int sampleN = 50;
                    int sampleP = 50;
                    for (int ind=0; (ind < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2) && (sampleN > 0 && sampleP > 0)) ; ind++)
                    {
                        int id = v[ind];
                    //while()
                        int n = neg + pos;
                        float accT = accText.at<float>(id);
                        
                        //printf("%f %f\n", accT,SPCTE->_arraySP[id].accLabel(1));
                        
                        if (( accT >= 0.5) && (pos < nSAMPLES))
                        {
                            descriptors.row(id).copyTo(trainingData.row(n));
                            labels.at<float>(n,0) = (float) LABEL_TEXT;
                            pos++;
                            sampleP--;
                            
                            string npos = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/pos/" + i->path().filename().string() + to_string(id) + ".png";
                            imwrite(npos,SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));
                            
                            
                        }else if (( accT == 0.0) && (neg < nSAMPLES))
                        {
                            descriptors.row(id).copyTo(trainingData.row(n));
                            labels.at<float>(n,0) = (float) LABEL_NOTEXT;
                            neg++;
                            sampleN--;
                            
                            string nneg = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/neg/" + i->path().filename().string() + to_string(id) + ".png";
                            imwrite(nneg,SPCTE->cropSuperpixel(SPCTE->getImageSuperpixels(),id,3));
                            
                        }
                        
                    }//for SPTCTE//*/

                
                    /*Mat desRGB,desLAB,desEDGES,desEDDIR,desCAFFE,desSEMANTIC,accText;
                    int num = SPCTE->maxID+1;
                    
                    bool readDescriptor = descriptorFileText(i->path().filename().string(), num,&desRGB,&desLAB,&desEDGES,&desEDDIR,&desCAFFE,&desSEMANTIC,&accText);
                    
                    //for each id in the image
                    for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                    {
                        
                        Mat desID;
                        
                        if (readDescriptor)
                            desID =  concatFileDescriptors(id,desRGB,desLAB,desEDGES, desEDDIR,  desCAFFE,  desSEMANTIC);//desSEMANTIC.row(id);
                        
                        else
                            desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen);
                        //Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                        
                        //ADD desID to SVM and labels
                        int n = neg + pos;
                        
                        if ((SPCTE->_arraySP[id].accLabel(1) >= 0.5) && (pos < nSAMPLES))
                        { //text
                           // Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                            labels.at<float>(n,0) = (float) LABEL_TEXT;
                            desID.row(0).copyTo(trainingData.row(n));
                            pos = pos + 1;
                            desID.release();
                        }
                        else
                        {
                            if ((SPCTE->_arraySP[id].accLabel(0) == 1.0) && (neg < nSAMPLES))
                            {
                               // Mat desID = descriptorText(SPCTE, id, nameEdges, nameSegmen);
                                labels.at<float>(n,0) = (float) LABEL_NOTEXT;
                                //add des in trainingData
                                desID.row(0).copyTo(trainingData.row(n));
                                neg = neg + 1;
                                desID.release();
                            }
                        }
                    }//for*/
                    
                    timeSuperpixels += SPCTE->timeSuperpixels;
                    timeLAB += SPCTE->timeLAB;
                    timeRGB += SPCTE->timeRGB;
                    timeEDGES += SPCTE->timeEDGES;
                    timeEDGESDIR += SPCTE->timeEDGESDIR;
                    timeCAFFE += SPCTE->timeCAFFE;
                    timeSEMANTIC += SPCTE->timeSEMANTIC;
                    
                    delete(SPCTE);
                    //descriptors.release();
                    //accText.release();
                }
             }
        }//if image
      }//for
    
    
    //TRAIN
    
    // Set up SVM's parameters
    //LINEAL
    CvSVMParams params;
    
    if (nameSVM.find("RBF") != std::string::npos)
    {
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::RBF;
        params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, (int)1e7, (double)1e-6);
        params.degree = 0;
        params.gamma = 5.383;
        params.coef0 = 0;
        
        params.C = 2.67; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        params.p = 0.0; // for CV_SVM_EPS_SVR
        
    }
   /* else  if (nameSVM.find("NU_SVC") != std::string::npos){
        
        params.svm_type    = CvSVM::NU_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   =  TermCriteria(CV_TERMCRIT_ITER, (int)1e7, (double)1e-6);
        params.nu = 0.5;
    }*/
    else
    {
        params.svm_type    = CvSVM::C_SVC;
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   =     TermCriteria(CV_TERMCRIT_ITER, (int)1e7, (double)1e-6);
    }
    
    // Train the SVM
    CvSVM SVM;
    
    clock_t startTrain = clock();
  
    SVM.train(trainingData, labels, Mat(), Mat(), params);
    SVM.save(nameSVM.c_str());
    SVM.clear();
    
    printf("\n===================================================================\n");
    printf("**** \tTRAIN %0.3f TOTAL %0.3f seconds\n",
           (float) (((double)(clock() - start)) / CLOCKS_PER_SEC),
           (float) (((double)(clock() - startTrain)) / CLOCKS_PER_SEC) );
    printf("===================================================================\n");
    printf("**** Superpixels: %f seconds\n ",timeSuperpixels);
    if (mLAB == 1)      printf("**** LAB: %f seconds\n ",timeLAB);
    if (mRGB == 1)      printf("**** RGB: %f seconds\n ",timeRGB);
    if (mEDGES == 1)    printf("**** EDGES: %f seconds\n ",timeEDGES);
    if (mEDDIR == 1)    printf("**** EDDIR: %f seconds\n ",timeEDGESDIR);
    if (mCAFFE == 1)    printf("**** CAFFE: %f seconds\n ",timeCAFFE);
    if (mSEMANTIC == 1) printf("**** SEMANTIC: %f seconds\n ",timeSEMANTIC);//*/
}

void calculateSUPERPIXELSdescriptors(SuperPixels *SPCTE, string nameEdges, string nameEdgesDIR, string nameSegmen, Mat *descriptors, Mat *accText, string path,string img)
{

    //READ????
    Mat desRGB,desLAB,desEDGES,desEDDIR,desCAFFE,desSEMANTIC;
    int num = SPCTE->maxID+1;
    
    // desSEMANTIC = Mat::zeros(num,2*(SEMANTIC_LABELS),CV_32FC1);*/
    bool readDescriptor = descriptorFileText(path,img, num,&desRGB,&desLAB,&desEDGES,&desEDDIR,&desCAFFE,&desSEMANTIC,accText);
    
    //for each id in the image
    for (int id=0; (id < SPCTE->maxID+1) ; id++)
    {
        //all des together
        //Mat desID = descriptorText(SPCTE, id, nameEdges,nameEdgesDIR, nameSegmen);
        Mat desID;
        
        if (readDescriptor)
            desID =  concatFileDescriptors(id,desRGB,desLAB,desEDGES, desEDDIR,  desCAFFE,  desSEMANTIC);//desSEMANTIC.row(id);
        
        else
            desID = descriptorText(SPCTE, id, nameEdges, nameEdgesDIR, nameSegmen);
        
        desID.row(0).copyTo(descriptors->row(id));
        
        float acc = SPCTE->_arraySP[id].accLabel(1);
        accText->at<float>(0,id) = acc;
        
    }//for superpixels
}

void descriptors2file(FILE *fout, float fid, Mat desID, float acc)
{
    fwrite(&fid,sizeof(float),1,fout);
    
    for(int d=0; d < desID.cols; d++)
        fwrite(&desID.at<float>(d),sizeof(float),1,fout);
    
    fwrite(&acc,sizeof(float),1,fout);
}

bool file2descriptors(string file, int size, Mat *descriptors, Mat *accText)
{
     FILE *fid = fopen(file.c_str(),"rb");
     if (fid == NULL)
         return false;
    
     int sizeID = size + 2;
     float data[sizeID];
     int id=0;
    
     while(!feof(fid))
     {
         int d = fread(data, sizeof(data[0]), sizeID, fid);
         //data: id des acc
         if (d > 0)
         {
         for (int i = 1; i <= size; i++)
         {
             descriptors->at<float>(id,i-1)= data[i];
             //printf("id: %d leidos: %d %f %f\n",id,d,data[i],descriptors->at<float>(id,i-1));
         }
             accText->at<float>(id)= data[size+1];
             id = id + 1;
         }
     }
    
    fclose(fid);
    return true;
}

void saveSUPERPIXELSdescriptors(SuperPixels *SPCTE, string img, string nameEdges, string nameEdgesDIR, string nameSegmen)
{
    img = "/Users/acambra/TESIS/CODE/GibHub_test_superpixels/build/Debug/ICDAR/test/descriptors/" + img;
    
    string fRGB = img + "_RGB" + ".bin";
    string fLAB = img + "_LAB" + ".bin";
    string fEDGES = img + "_EDGES" + ".bin";
    string fEDDIR = img + "_EDDIR" + ".bin";
    string fCAFFE = img + "_CAFFE" + ".bin";
    string fSEMANTIC = img + "_SEMANTIC" + ".bin";
    
    FILE* frgb,*flab,*fedges,*feddir,*fcaffe,*fsemantic;
    
    frgb = fopen(fRGB.c_str(), "wb");
    flab = fopen(fLAB.c_str(), "wb");
    fedges = fopen(fEDGES.c_str(), "wb");
    feddir= fopen(fEDDIR.c_str(), "wb");
    fcaffe = fopen(fCAFFE.c_str(), "wb");
    fsemantic = fopen(fSEMANTIC.c_str(),"wb");
    
    //EDGES
    Mat imgEdges = imread(nameEdges,CV_LOAD_IMAGE_COLOR);
    Mat imgEdgesDIR  = loadMatFromYML(nameEdgesDIR,"N");

    //SEMANTIC
    Mat imgSegmen  = imread(nameSegmen,CV_LOAD_IMAGE_GRAYSCALE);//COLOR

    Mat desID;
    //for each id in the image
    for (int id=0; (id < SPCTE->maxID+1) ; id++)
    {
       
        float fid= (float) id;
        float acc = SPCTE->_arraySP[id].accLabel(1);
        
        //LAB: ID DES ACC
        desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,1,NBINS_L,NBINS_AB \
                                                ,0,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS,imgSegmen).clone();
        descriptors2file(flab, fid, desID, acc);
        
        //RGB: ID DES ACC
        desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                                ,0,NBINS_L,NBINS_AB \
                                                ,1,NBINS_RGB\
                                                ,0,NBINS_PEAKS \
                                                ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                                ,0,NBINS_EDDIR,imgEdgesDIR\
                                                ,0,CAFFE_LAYER,NUMCAFFE\
                                                ,0,SEMANTIC_LABELS,imgSegmen).clone();
        descriptors2file(frgb, fid, desID, acc);
        
        //EDGES: ID DES ACC
        desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                            ,0,NBINS_L,NBINS_AB \
                                            ,0,NBINS_RGB\
                                            ,0,NBINS_PEAKS \
                                            ,1,NBINS_EDGES,modeEDGES,imgEdges\
                                            ,0,NBINS_EDDIR,imgEdgesDIR\
                                            ,0,CAFFE_LAYER,NUMCAFFE\
                                            ,0,SEMANTIC_LABELS,imgSegmen).clone();
        descriptors2file(fedges, fid, desID, acc);
        
        //EDDIR: ID DES ACC
        desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                            ,0,NBINS_L,NBINS_AB \
                                            ,0,NBINS_RGB\
                                            ,0,NBINS_PEAKS \
                                            ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                            ,1,NBINS_EDDIR,imgEdgesDIR\
                                            ,0,CAFFE_LAYER,NUMCAFFE\
                                            ,0,SEMANTIC_LABELS,imgSegmen).clone();
        descriptors2file(feddir, fid, desID, acc);
        
        //SEMANTIC: ID DES ACC
        desID = SPCTE->calculateDescriptors(id,SPCTE->getImage() \
                                            ,0,NBINS_L,NBINS_AB \
                                            ,0,NBINS_RGB\
                                            ,0,NBINS_PEAKS \
                                            ,0,NBINS_EDGES,modeEDGES,imgEdges\
                                            ,0,NBINS_EDDIR,imgEdgesDIR\
                                            ,0,CAFFE_LAYER,NUMCAFFE\
                                            ,1,SEMANTIC_LABELS,imgSegmen).clone();
        descriptors2file(fsemantic, fid, desID, acc);
        
        //CAFFE
        Mat imageSP = SPCTE->cropSuperpixel(SPCTE->getImage(),id,1).clone();
        Mat desCaf= _caffe->features(imageSP, "fc7").clone();
        normalize(desCaf, desCaf);
        
        descriptors2file(fcaffe, fid, desCaf, acc);
        
    }//for superpixels
    
    fclose(frgb);
    fclose(flab);
    fclose(fedges);
    fclose(feddir);
    fclose(fcaffe);
    fclose(fsemantic);
    
    printf("\tSaved descriptors: %s\n",img.c_str());
    
    delete(SPCTE);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////// CALCULAR!

/*  string fileDesImg = /*dir_des +*/ /*nameSVM.substr(nameSVM.find("_"), nameSVM.find_last_of(".")) + "_" + i->path().filename().string() + ".bin";
                                       fout = fopen(fileDesImg.c_str(),"wb");
                                       
                                       
                                       //for each id in the image
                                       for (int id=0; (id < SPCTE->maxID+1 && (neg+pos) < (nSAMPLES*2)) ; id++)
                                       {
                                       
                                       Mat desID = descriptorText(SPCTE, id, nameEdges,nameEdgesDIR, nameSegmen);
                                       
                                       // fprintf(fout,"%d\n",id);
                                       float fid= (float) id;
                                       fwrite(&fid,sizeof(float),1,fout);
                                       //printf("--->%f\n",fid);
                                       //
                                       for(int d=0; d < desID.cols; d++)
                                       {
                                       //fprintf(fout,"%0.6f ",desID.at<float>(d));
                                       //fwrite(desID.data,sizeof(float),1,desID.cols,fout);
                                       fwrite(&desID.at<float>(d),sizeof(float),1,fout);
                                       // printf("--->%f\n",desID.at<float>(d));
                                       
                                       }
                                       //fwrite(&labels.at<float>(n,0),sizeof(float),1,fout);
                                       float acc = SPCTE->_arraySP[id].accLabel(1);
                                       fwrite(&acc,sizeof acc,1,fout);
                                       //ADD desID to SVM and labels
                                       int n = neg + pos;
                                       
                                       if ((SPCTE->_arraySP[id].accLabel(1) >= 0.5) && (pos < nSAMPLES))
                                       { //text
                                       Mat desID = descriptorText(SPCTE, id, nameEdges,nameEdgesDIR, nameSegmen);
                                       labels.at<float>(n,0) = (float) LABEL_TEXT;
                                       desID.row(0).copyTo(trainingData.row(n));
                                       pos = pos + 1;
                                       desID.release();
                                       
                                       // fprintf(fout,"%d %0.4f ",LABEL_TEXT,SPCTE->_arraySP[id].accLabel(1));
                                       /*fwrite(&labels.at<float>(n,0),sizeof(float),1,fout);
                                       float acc = SPCTE->_arraySP[id].accLabel(1);
                                       fwrite(&acc,sizeof acc,1,fout);*/

/*  }
 else
 {
 if ((SPCTE->_arraySP[id].accLabel(0) == 1.0) && (neg < nSAMPLES))
 {
 Mat desID = descriptorText(SPCTE, id, nameEdges,nameEdgesDIR, nameSegmen);
 labels.at<float>(n,0) = (float) LABEL_NOTEXT;
 //add des in trainingData
 desID.row(0).copyTo(trainingData.row(n));
 neg = neg + 1;
 desID.release();
 //fprintf(fout,"%d %0.4f ",LABEL_NOTEXT,SPCTE->_arraySP[id].accLabel(0));
 /*fwrite(&labels.at<float>(n,0),sizeof(float),1,fout);
 float acc = SPCTE->_arraySP[id].accLabel(0);
 fwrite(&acc,sizeof acc,1,fout);*/
/*   }
 
 
 }
 
 // fprintf(fout,"%d\n",id);//,i->path().filename().string().c_str());
 //desID.release();
 }//for superpixels
 
 fclose(fout);
 
 /*fout = fopen(fileDesImg.c_str(),"rb");
 float data[26];
 
 
 while(!feof(fout))
 {
 fread(data, sizeof(data[0]), 26, fout);
 
 
 for (int i = 0; i < 26; i++)
 printf("%0.6f ", data[i]);
 
 getchar();
 }*/


/*timeSuperpixels += SPCTE->timeSuperpixels;
 timeLAB += SPCTE->timeLAB;
 timeRGB += SPCTE->timeRGB;
 timeEDGES += SPCTE->timeEDGES;
 timeEDGESDIR += SPCTE->timeEDGESDIR;
 timeCAFFE += SPCTE->timeCAFFE;
 timeSEMANTIC += SPCTE->timeSEMANTIC;//*/

/*  delete(SPCTE);
 }//if
 }*/
