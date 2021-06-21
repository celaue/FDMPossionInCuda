#include "EquationSystem/EquationSystem.hpp"
#include "Plotter/Plotter.hpp"
#include <iostream>
#include <ctime>


int main(){
    int n_grid = 30;
    std::clock_t start;
    double duration;
    EquationSystem solver(n_grid,1);
    solver.init_System();

    start = std::clock();
    solver.get_solution("Jacobiparallel");
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Cuda solver: "<< duration <<'\n';

    
    start = std::clock();
    solver.get_solution("Jacobisequential");
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Not solver: "<< duration <<'\n';

    // start = std::clock();
    // solver.get_solution(false);
    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // std::cout<<"Eigen solver: "<< duration <<'\n';

    std::vector<double> x_data,y_data,z_data;
    solver.write_to_csv();
    solver.write_to_vector(x_data,y_data,z_data);
    myPlotter plotter(z_data,n_grid,n_grid);
    mglQT gr(&plotter, "Solution");
    return gr.Run();
    // return my_plot_surface_solution(x_data,y_data,z_data,30,30);
}