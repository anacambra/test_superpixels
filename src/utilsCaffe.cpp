//
//  utilsCaffe.cpp
//  test_sup
//
//  Created by Ana B. Cambra on 09/02/16.
//
//

#include <string>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"


using namespace caffe;
using namespace std;
using namespace cv;

//using namespace boost;

using NetSp = boost::shared_ptr<caffe::Net<float>>;
//using BlobSp = boost::shared_ptr<Blob<float>>;
/* using SolverSp = std::shared_ptr<caffe::Solver<float>>;*/
//using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;


class utilsCaffe
{
    caffe::Net<float> *net;
    
public:
    
    utilsCaffe(string model, string proto)
    {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        
        net = new  caffe::Net<float>(proto,caffe::TEST);
        net->CopyTrainedLayersFrom(model);
        
        //check size data layer
        /*const boost::shared_ptr<Blob<float>> data = net->blob_by_name("data");
         printf("params %s; num: %d chanels: %d widht: %d height: %d\n", data->shape_string().c_str(),
         net->blobs()[0]->shape(0),net->blobs()[0]->shape(1),net->blobs()[0]->shape(2),net->blobs()[0]->shape(3));getchar();*/
        /*dim:  10  data->shape(0); num
         dim:   3   data->shape(1); channels
         dim:   227 data->shape(2); width
         dim:   227 data->shape(3); height
         */
        
    }
    
    ~utilsCaffe()
    {
      // if (net != NULL)
           delete net;
    }
    
    
    Mat features(Mat img, string layer)
    {
        Mat new_mat;
        
        int num = net->blobs()[0]->shape(0);
        int channels = net->blobs()[0]->shape(1);
        int width = net->blobs()[0]->shape(2);
        int height = net->blobs()[0]->shape(3);
        //reshape img
        if (img.cols != width || img.rows != height)
        {
            resize(img, new_mat, Size(width, height),0,0,CV_INTER_LINEAR);
        }else
            new_mat = img.clone();
        
        
        // convert the image to a caffe::Blob
        caffe::Blob<float> * blob = new caffe::Blob<float>(num, channels, width, height);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < new_mat.rows; ++h) {
                for (int w = 0; w < new_mat.cols; ++w) {
                    blob->mutable_cpu_data()[blob->offset(0, c, h, w)] = new_mat.at<cv::Vec3b>(h, w)[c];
                }
            }
        }

        // create input and output vectors for the Forward call
        vector<caffe::Blob<float>*> bottom_vecs;
        vector<caffe::Blob<float>*> top_vecs;
        bottom_vecs.push_back(blob);
        
        //solve net
        top_vecs = net->Forward(bottom_vecs);
        
        //OBTAIN LAYER
        const boost::shared_ptr<Blob<float>> features = net->blob_by_name(layer.c_str());
        
        Mat descriptor =  Mat::zeros(1, features->channels(), CV_32FC1);
        
        //printf("SIZES %s: %d %d %d \n",layer.c_str(),features->channels(),features->height(),features->width());
        int i=0;
        for (int c = 0; c < features->channels(); ++c) { //TODO! en este ejemplo da igual el channel!!!!!!! OJO!!!!!
            for (int h = 0; h < features->height(); ++h) {
                for (int w = 0; w < features->width(); ++w) {
                    descriptor.at<float>(0,c) = features->data_at(0, c, h, w); i++;
                    //printf("%d %d %d: %f\n",c, h, w,features->data_at(0, c, h, w));i++;
                }
            }
        }
        
        delete blob;
        new_mat.release();
        
        return descriptor;
    }
    
    int classifier(Mat img)
    {
        Mat new_mat;
        
        int num = net->blobs()[0]->shape(0);
        int channels = net->blobs()[0]->shape(1);
        int width = net->blobs()[0]->shape(2);
        int height = net->blobs()[0]->shape(3);
        //reshape img
        if (img.cols != width || img.rows != height)
        {
            resize(img, new_mat, Size(width, height),0,0,CV_INTER_LINEAR);
        }else
            new_mat = img.clone();
        
        
        // convert the image to a caffe::Blob
        caffe::Blob<float> * blob = new caffe::Blob<float>(num, channels, width, height);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < new_mat.rows; ++h) {
                for (int w = 0; w < new_mat.cols; ++w) {
                    blob->mutable_cpu_data()[blob->offset(0, c, h, w)] = new_mat.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
        
        // create input and output vectors for the Forward call
        vector<caffe::Blob<float>*> bottom_vecs;
        vector<caffe::Blob<float>*> top_vecs;
        bottom_vecs.push_back(blob);
        
        //solve net
        top_vecs = net->Forward(bottom_vecs);
        
        // calculate mean of the resulting blob, which is the classification result?
        caffe::Blob<float>* topvec = top_vecs[0];
        float mean = 0;
        for (int c = 0; c < topvec->channels(); ++c) {
            for (int h = 0; h < topvec->height(); ++h) {
                for (int w = 0; w < topvec->width(); ++w) {
                    mean += topvec->data_at(0, c, h, w);
                }
            }
        }
        mean /= (topvec->channels() * topvec->height() * topvec->width());
        
        
        new_mat.release();
        return mean;
        
    }
};

