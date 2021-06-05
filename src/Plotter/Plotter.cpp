#include "Plotter.hpp"

int myPlotter::Draw(mglGraph *gr){
    
    gr->Title("Solution of Poisson Equation");	
    gr->Box(); gr->SetRange('z',-1,1);gr->Surf(prepare_for_drawing(),"BbwrR");
    return 0;
}


mglData myPlotter::prepare_for_drawing()
{
    mglData a;
    a.Create(nx_,ny_); 
    
    int i0 =0;
    for(int i=0;i<nx_;i++)  for(int j=0;j<ny_;j++)
    { 
        i0 = i+nx_*j;
        a.a[i0] = z_data_[i0]/(*std::max_element(z_data_.begin(),z_data_.end()))-0.5;
    }
    return a;
}
