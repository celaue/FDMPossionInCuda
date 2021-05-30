#include "System/System.hpp"
#include <iostream>


int main(){
    EquationSystem solver(80,1);
    solver.init_System();
    std::vector<double> result = solver.get_solution();
    solver.write_to_csv(result);
}