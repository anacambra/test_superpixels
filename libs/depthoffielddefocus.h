//
//  main.cpp
//  test_superpixels
//
//  Created by Ana Cambra on 26/11/14.
//  Copyright (c) 2014 Ana Cambra. All rights reserved.
//

//#include <iostream>
//Opencv
/*#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"*/
//#include <chrono>
//#include <strings.h>

/**
 *
 * maxBlur in number of pixels
 **/
std::vector<cv::Mat> create_blurred(const cv::Mat& input, double maxBlur, int nbins)
{
	std::vector<cv::Mat> blurred(nbins);
	blurred[0] = input;
	for (int i = 1; i<nbins;++i)
	{
		double blur = double(i)*maxBlur/double(nbins);
		int blur_size = int(blur);
		if (blur_size < 1) blur_size = 1;
		else if ((blur_size % 2) == 0) ++blur_size;
 		cv::GaussianBlur(input, blurred[i], cv::Size(blur_size,blur_size), 0);
	//	cv::imwrite("blurred_"+std::to_string(i)+".jpg",blurred[i]);
	}

	return blurred;
}


cv::Mat interpolate_in_vector(const std::vector<cv::Mat>& source,
		const cv::Mat& interpolant)
{
    cv::Mat sol = cv::Mat::ones(source[0].rows, source[0].cols, source[0].type()); 
    cv::Mat_<cv::Vec3b> sol_  = sol;
    
    for( int i = 0; i < interpolant.rows; ++i) for( int j = 0; j < interpolant.cols; ++j )
    {
	float d = interpolant.at<float>(i,j);
	int   layer = trunc(d*float(source.size() - 1)); 
	if (layer >= (source.size()-1)) sol_(i,j) = source[source.size()-1].at<cv::Vec3b>(i,j);
	else {
		float t = std::min((d*float(source.size() - 1)) - float(layer), 1.0f);
		sol_(i,j) = source[layer  ].at<cv::Vec3b>(i,j)*(1.0f - t) +
		    	    source[layer+1].at<cv::Vec3b>(i,j)*t;
	}
    }

    sol = sol_;
    return sol;
}

std::string type2str(int type) {
	std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

float blur_size_from_distance(float distance, float focal_distance, float focal_length, float aperture, bool linear)
{
	if (linear) return aperture*std::abs(distance - focal_distance)/focal_length;
	else return 
		aperture*(255.0 - focal_length)*std::abs(distance - focal_distance)/((255.0f - distance)*(focal_length - focal_distance));
}


/**
 * Esto no lo usamos ya, no son layers sino iteraciones hasta la solucion, así que
 * hemos copiado y pegado el código
 **/
std::vector<cv::Mat> create_depth_layers_old(const cv::Mat& input, const cv::Mat& depth, 
		int nbins, float focal_distance, float focal_length, float aperture)

{
        std::vector<cv::Mat> layers(nbins);
	float ddepth = 255.0f/float(nbins);
	cv::Mat acc_mask;
	for (int i = 0; i<nbins;++i)
	{
		cv::Mat mask = (depth <= (float(nbins - i)*ddepth));
		cv::cvtColor(mask, mask, CV_GRAY2RGB);
		cv::Mat masked;
		cv::bitwise_and(input, mask, masked);
		mask.convertTo(mask, CV_32F, 1.0f/255.0f);
		double blur = blur_size_from_distance((float(nbins - i - 1) + 0.5f)*ddepth,
				focal_distance, focal_length, aperture, true);
		int blur_size = int(blur);
		//std::cerr<<i<<" - "<<blur_size<<std::endl;
		if (blur_size >= 1) {
			if ((blur_size % 2) == 0) ++blur_size;
 			cv::GaussianBlur(masked, masked, cv::Size(blur_size,blur_size), 0);
 			cv::GaussianBlur(mask,     mask, cv::Size(blur_size,blur_size), 0);
		}
		cv::imwrite("masked_"+std::to_string(i)+".jpg",masked);
		if (i == 0) {
			layers[i] = masked.clone(); 
			acc_mask  = mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f); 
		} else {
			cv::max(acc_mask, mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f), acc_mask);
			cv::multiply(layers[i-1], acc_mask , layers[i], 1, CV_8U);
			cv::divide  (masked,      mask, masked        , 1, CV_8U);
			cv::multiply(masked, (acc_mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f)), masked   , 1, CV_8U);
		        layers[i] += masked;	
		}
		cv::imwrite("layer_"+std::to_string(i)+".jpg",layers[i]);
		
	}

	return layers;
}

/**
 * Esto está en progreso, no funciona
 **/
std::vector<cv::Mat> create_depth_layers_with_interpolation(const cv::Mat& input, const cv::Mat& depth, 
		int nbins, float focal_distance, float focal_length, float aperture)

{
        std::vector<cv::Mat> layers(nbins);
	cv::Mat acc_mask;
	for (int i = 0; i<nbins;++i)
	{
		float bin_depth      = float(nbins - i - 1)*255.0f/float(nbins - 1);
		cv::Mat mask = abs(depth - bin_depth)*float(nbins - 1)/255.0f;
		cv::min(mask, 1.0f, mask);
		mask *= -1.0f;
		mask +=  1.0f;
		cv::cvtColor(mask, mask, CV_GRAY2RGB);
		cv::Mat masked;
		cv::multiply(input,      mask, masked        , 1, CV_8U);
		double blur = blur_size_from_distance(bin_depth,
				focal_distance, focal_length, aperture, true);
		int blur_size = int(blur);
		if (blur_size >= 1) {
			if ((blur_size % 2) == 0) ++blur_size;
 			cv::GaussianBlur(masked, masked, cv::Size(blur_size,blur_size), 0);
 			cv::GaussianBlur(mask,     mask, cv::Size(blur_size,blur_size), 0);
		}
		if (i == 0) {
			layers[i] = masked.clone(); 
			acc_mask  = mask; 
		} else {
			acc_mask += mask;
			cv::min(acc_mask, cv::Scalar(1.0f,1.0f,1.0f), acc_mask);
//			cv::multiply(layers[i-1], acc_mask , layers[i], 1, CV_8U);
			cv::divide  (masked,      mask, masked        , 1, CV_8U);
			cv::multiply(masked, (acc_mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f)), masked   , 1, CV_8U);
		        layers[i] = layers[i-1] + masked;	
		}
	}

	return layers;
}


//Blur image
cv::Mat blur_image_depth(const cv::Mat& image, const cv::Mat& depth, 
		int nbins, float focal_distance, float focal_length, float aperture, bool linear) {
        
	std::chrono::time_point<std::chrono::system_clock> start;
        start = std::chrono::system_clock::now();
        cv::Mat sol;
        float ddepth = 255.0f/float(nbins);
	cv::Mat acc_mask;
	for (int i = 0; i<nbins;++i)
	{
		cv::Mat mask = (depth <= (float(nbins - i)*ddepth));
		cv::cvtColor(mask, mask, CV_GRAY2RGB);
		cv::Mat masked;
		cv::bitwise_and(image, mask, masked);
		mask.convertTo(mask, CV_32F, 1.0f/255.0f);
		double blur = blur_size_from_distance((float(nbins - i - 1) + 0.5f)*ddepth,
				focal_distance, focal_length, aperture, linear);
		int blur_size = int(blur);
		if (blur_size >= 1) {
			if ((blur_size % 2) == 0) ++blur_size;
 			cv::GaussianBlur(masked, masked, cv::Size(blur_size,blur_size), 0);
 			cv::GaussianBlur(mask,     mask, cv::Size(blur_size,blur_size), 0);
		}
		//std::cout<<"Layer "<<i<<" - Depth "<<((float(nbins - i - 1) + 0.5f)*ddepth)<<
		//			" - Blur "<<blur_size<<std::endl;
		if (i == 0) {
			sol = masked.clone();
			acc_mask  = mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f); 
		} else {
			acc_mask  += mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f);
			cv::min(acc_mask, cv::Scalar(1.0f,1.0f,1.0f), acc_mask);
			cv::multiply(sol, acc_mask , sol, 1, CV_8U);
			cv::divide  (masked,      mask, masked        , 1, CV_8U);
			cv::multiply(masked, (acc_mask*(-1.0f) + cv::Scalar(1.0f,1.0f,1.0f)), masked   , 1, CV_8U);
		        sol += masked;	
		}
	}

   // std::cerr<<"Tiempo blur depth : "<<std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()<<std::endl;
    return sol;
}


//Blur image
cv::Mat blur_image_focal_distance(const cv::Mat& image, const cv::Mat& depth, 
		int nbins, float focal_distance, float focal_length, float aperture, bool linear)
{
    std::chrono::time_point<std::chrono::system_clock> start;

    start = std::chrono::system_clock::now();
    auto blurred = create_blurred(image, aperture*255.0/focal_length, nbins);
    
    cv::Mat distance_to_focus = cv::abs(depth - focal_distance);
    double min, max;
    cv::minMaxLoc(distance_to_focus, &min, &max);
    auto sol = interpolate_in_vector(blurred, distance_to_focus/max);  

    //std::cerr<<"Tiempo blur : "<<std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()<<std::endl;
    return sol;
}


void  adolfoBlur() {
	cv::String nimage  ="cocacola.jpg";
	cv::String ndepth  ="cocacola_depth.png";
	cv::String nresult ="blurred.jpg";

    int   nbins          = 8;
    float aperture       = 7.0;
    float focal_distance = 205.0;
    float focal_length   = 127.0;
    bool  linear         = true;

   /* for (int i = 1;i<argc;++i)
    {
    if (i<argc - 1) {
	if (strcmp("-input",argv[i])==0)               nimage         = argv[++i];
	else if (strcmp("-depth",argv[i])==0)          ndepth         = argv[++i];
	else if (strcmp("-output",argv[i])==0)         nresult        = argv[++i];
	else if (strcmp("-nbins",argv[i])==0)          nbins          = atoi(argv[++i]);
	else if (strcmp("-aperture",argv[i])==0)       aperture       = atof(argv[++i]);
	else if (strcmp("-focal-distance",argv[i])==0) focal_distance = atof(argv[++i]);
	else if (strcmp("-focal-length",argv[i])==0)   focal_length   = atof(argv[++i]);
    }
	if (strcmp("-real-camera",argv[i])==0)            linear = false;
	else if (strcmp("-linear-camera",argv[i])==0)     linear = true;
        
    }*/
 

    cv::Mat image = cv::imread(nimage,CV_LOAD_IMAGE_COLOR);
    
    cv::Mat depth = cv::imread(ndepth,CV_LOAD_IMAGE_GRAYSCALE);
    depth.convertTo(depth,CV_32FC1);
    cv::normalize(depth, depth, 0.01, 1.0, cv::NORM_MINMAX, -1);
    depth = depth * 255.0;   
    
    cv::Mat f1= blur_image_depth(image, depth, nbins,focal_distance,focal_length,aperture, linear);
   
    cv::imwrite(nresult, f1);
    
    imshow("final",f1);
}
