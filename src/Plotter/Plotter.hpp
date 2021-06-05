#ifndef DRAWING_HPP
#define DRAWING_HPP

#include <iostream>
#include <mgl2/qt.h>
#include <vector>
#include <algorithm>


class myPlotter : public mglDraw {
public:
    myPlotter(std::vector<double> z_data,int nx, int ny) : z_data_{z_data},nx_{nx},ny_{ny}{}
    int Draw(mglGraph *gr) override;
private:
    mglData prepare_for_drawing();
    std::vector<double> z_data_;
    int nx_,ny_;
};


#endif