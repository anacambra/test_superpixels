#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <set>

//Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

class SuperPixel
{

    int _id;
    //mascara de la imagen original
    Mat _mask;
    int _numPixels;
    int _label;
    
    //neighbour ids
    set<int> _neighbours;
    
    
/*    MatND hist_l;
    float l_median;
    float l_mean;
    float l_std;
    
    MatND hist_a;
    float a_median;
    float a_mean;
    float a_std;
    
    MatND hist_b;
    float b_median;
    float b_mean;
    float b_std;*/
    
    
    
   /* MatND hist_depth;
    //mediana sin ceros
    float d_median;
    //media
    float d_mean;
    //varianza
    float d_var;
    //precision
    float d_acc;
    float depth;*/
    
    
    public:
    
    SuperPixel(){ _id = -1; _numPixels = 0; _mask = Mat::zeros(0, 0, 0);  _label = -1; }
    ~SuperPixel(){} ;
    
    void initialize(int id,Mat mask, int label){ _id = id; _mask = mask.clone(); _numPixels = countNonZero(mask); _label = label;}
    int getId(){ return _id;}
    int getNumPixels(){ return _numPixels;}
    int getLabel(){ return _label;}
    Mat getMask(){ return _mask;}
    
    set<int> getNeighbours(){ return _neighbours;}    
    void addNeighbour(int n){ if (_id != n) {_neighbours.insert(n);printf("id: %d  v: %d \n",_id, n);}}
    
    string toString(){ return "ID: " + to_string(_id) + " numPixels: " +to_string(_numPixels) + " label: " + to_string(_label);}
    
    
    
    /***************/
    //DESCRIPTORS
    /***************/
    //meanColor RGB
    //histogram color RGB, LAB, edges
    
    //VECINOS

};

#endif // SUPERPIXEL_H
