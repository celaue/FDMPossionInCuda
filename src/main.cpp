#include "Equation_System/Equation_System.hpp"
#include "Plotter/Plotter.hpp"
#include <iostream>
#include <ctime>


int main(){
    std::clock_t start;
    double duration;
    EquationSystem solver(60,1);
    solver.init_System();

    start = std::clock();
    solver.get_solution(true);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Cuda solver: "<< duration <<'\n';

    // start = std::clock();
    // solver.get_solution(false);
    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // std::cout<<"Eigen solver: "<< duration <<'\n';

    std::vector<double> x_data,y_data,z_data;
    solver.write_to_csv();
    solver.write_to_vector(x_data,y_data,z_data);
    myPlotter plotter(z_data,60,60);
    mglQT gr(&plotter, "Solution");
    return gr.Run();
    // return my_plot_surface_solution(x_data,y_data,z_data,30,30);
}