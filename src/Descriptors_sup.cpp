//
//  Descriptors_sup.cpp
//  
//
//  Created by Ana B. Cambra on 18/10/16.
//
//

#include <stdio.h>
#include "superpixels.cpp"
#include "Descriptors.cpp"

/////////////
Mat calculateDescriptors(int id, Mat image,
                         int mLAB   = 0, int NBINS_L     = 7, int NBINS_AB=7,
                         int mRGB   = 0, int NBINS_RGB   = 256,
                         int mPEAKS = 0, int NBINS_PEAKS = 64,
                         int mEDGES = 0, int NBINS_EDGES = 8, int modeEDGES = 0, Mat edges = Mat::zeros(1,1,CV_32FC3),
                         int mEDDIR = 0, int NBINS_EDDIR = 8, Mat edgesDIR = Mat::zeros(1,1,CV_32FC3),
                         int mCAFFE = 0, string CAFFE_LAYER = "fc7", int NUMCAFFE = 4096,
                         int mSEMANTIC = 0, int SEMANTIC_LABELS = 60,
                         int mCONTEXT = 0, int CONTEXT_LABELS = 60,
                         int mCONTEXT2 = 0, int CONTEXT2_LABELS = 60,
                         int mGLOBAL = 0, int GLOBAL_LABELS = 60,
                         int mCONLAB = 0,
                         int mCONEDGES = 0,
                         int mCONEDDIR = 0,
                         int mHOG =0,   int mCONHOG = 0)
{
    Mat des;
    
    if (mLAB != 0)
    {
        Mat imageSP = cropSuperpixel(image,id,1,BB_SP_MASK).clone();
        des=Descriptors::descriptorsLAB(imageSP,Mat(),NBINS_L,NBINS_AB).clone();
    }
    
    if (mCONLAB != 0)
    {
        
        if (des.rows != 0)
            hconcat(Descriptors::descriptorsLAB(image,getMaskNeigboursBB(id),NBINS_L,NBINS_AB), des,des);
        else
            des=Descriptors::descriptorsLAB(image,getMaskNeigboursBB(id),NBINS_L,NBINS_AB).clone();
        
    }
    
    if (mRGB != 0)
    {
        if (des.rows != 0)
            hconcat(Descriptors::descriptorsRGB(image,getMask(id),NBINS_RGB), des,des);
        else
            des=Descriptors::descriptorsRGB(image,getMask(id),NBINS_RGB).clone();
    }
    
    if (mPEAKS != 0)
    {
        if (des.rows != 0)
            hconcat(Descriptors::descriptorsPEAKS(image,getMask(id),NBINS_PEAKS), des,des);
        else
            des=Descriptors::descriptorsPEAKS(image,getMask(id),NBINS_PEAKS).clone();
    }
    
    if (mEDGES != 0)
    {
        clock_t start = clock();
        
        if (des.rows != 0)
        {
            Mat imageSP = cropSuperpixel(edges,id,1,BB_SP_MASK).clone();
            hconcat(Descriptors::descriptorsEDGES(imageSP,Mat(),NBINS_EDGES,modeEDGES), des,des);
        }
        else
        {
            Mat imageSP = cropSuperpixel(edges,id,1,BB_SP_MASK).clone();
            des=Descriptors::descriptorsEDGES(imageSP,Mat(),NBINS_EDGES,modeEDGES);
            //_arraySP[id].descriptorsEDGES(edges,NBINS_EDGES,modeEDGES).clone();
        }
        timeEDGES += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
    }
    
    if (mCONEDGES != 0)
    {
        
        if (des.rows != 0)
        {
            hconcat(Descriptors::descriptorsEDGES(edges,getMaskNeigboursBB(id),NBINS_EDGES,modeEDGES), des,des);
        }
        else
        {
            des=Descriptors::descriptorsEDGES(edges,getMaskNeigboursBB(id),NBINS_EDGES,modeEDGES);
        }
    }
    
    
    if (mEDDIR != 0)
    {
        clock_t start = clock();
        if (des.rows != 0)
            hconcat(Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskBB(id),NBINS_EDDIR), des,des);
        else
            des=Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskBB(id),NBINS_EDDIR).clone();
        timeEDGESDIR += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
    }
    
    if (mCONEDDIR != 0)
    {
        clock_t start = clock();
        if (des.rows != 0)
            hconcat(Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskNeigboursBB(id),NBINS_EDDIR), des,des);
        else
            des=Descriptors::descriptorsEDGESDIR(edges,edgesDIR,getMaskNeigboursBB(id),NBINS_EDDIR).clone();
        timeEDGESDIR += (float) (((double)(clock() - start)) / CLOCKS_PER_SEC);
    }
    
    //HOG
    if (mHOG != 0)
    {
        //get image BB superpixel
        Mat imageSP = cropSuperpixel(image,id,1,BB_SP_MASK).clone();
        
        if (des.rows != 0)
            hconcat(Descriptors::DescriptorHOG(imageSP), des,des);
        else
            des=Descriptors::DescriptorHOG(imageSP).clone();
        
    }
    if (mCONHOG != 0)
    {
        Mat imageSP = cropSuperpixel(image,id,1,BB_NEIG_MASK).clone();
        
        if (des.rows != 0)
            hconcat(Descriptors::DescriptorHOG(imageSP), des,des);
        else
            des=Descriptors::DescriptorHOG(imageSP).clone();
    }
    
    if (mSEMANTIC != 0)
    {
        if (des.rows != 0)
            hconcat(_arraySP[id].descriptorsSEMANTIC(SEMANTIC_LABELS),des,des);
        else
            des=_arraySP[id].descriptorsSEMANTIC(SEMANTIC_LABELS).clone();
        
    }
    
    if (mCONTEXT != 0)
    {
        if (des.rows != 0)
            hconcat(_arraySP[id].descriptorsCONTEXT(CONTEXT_LABELS),des,des);
        else
            des=_arraySP[id].descriptorsCONTEXT(CONTEXT_LABELS).clone();
        
    }
    
    if (mCONTEXT2 != 0)
    {
        if (des.rows != 0)
            hconcat(_arraySP[id].descriptorsCONTEXT_ORIENTED(CONTEXT2_LABELS),des,des);
        else
            des=_arraySP[id].descriptorsCONTEXT_ORIENTED(CONTEXT2_LABELS).clone();
        
    }
    
    if (mGLOBAL != 0)
    {
        if (des.rows != 0)
            hconcat(_arraySP[id].descriptorsCONTEXT_GLOBAL(GLOBAL_LABELS),des,des);
        else
            des=_arraySP[id].descriptorsCONTEXT_GLOBAL(GLOBAL_LABELS).clone();
        
    }
    
    return des.clone();
}
