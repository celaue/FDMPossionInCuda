#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <iostream>
#include <Eigen/Dense>
#include<Eigen/IterativeLinearSolvers>	
#include <unordered_map>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <cmath>
#include <fstream>
//L*x = b
struct SystemObjects{
    void add(std::tuple<std::set<int>,double> obj){
        obj_pos_list.push_back(std::get<0>(obj));
        obj_value_list.push_back(std::get<1>(obj));
    }
    std::vector<std::set<int>> obj_pos_list;
    std::vector<double> obj_value_list;
};
class EquationSystem{
    public:
        
        EquationSystem(int size,int length) {     // Constructor
            size_ = size;
            length_ = length;
            dx_ = length_/(size_-1);
            m_size_ = size_*size_;
            L_gen_ = Eigen::MatrixXd::Zero(m_size_, m_size_);
            b_ = Eigen::VectorXd::Zero(m_size_);
            x_b_ = Eigen::VectorXd::Zero(m_size_);
            boundary_ = std::vector<bool>(m_size_);
            std::fill(boundary_.begin(), boundary_.end(), false);
            generate_init_L();
            std::cout<<"System params: \n"<<"size: "<<size_<<" length: "<<length_<<" delta: "<<dx_<<"\n";
        }
    
        void init_System(){
            generate_equation_system();
        }
        std::vector<double> get_solution();
        void write_to_csv(std::vector<double> x);
        
    private:
        void generate_equation_system();
        void generate_init_L();
        
        //Utility
        std::tuple<std::set<int>,double> get_line(double x_start, double y_start,double x_end, double y_end,double value);
        std::tuple<std::set<int>,double> get_filled_rect(double x1, double y1,double x2, double y2,double value);
        std::tuple<std::set<int>,double> get_empty_rect(double x1, double y1,double x2, double y2,double value);
        std::tuple<std::set<int>,double> get_filled_circ(double m_x, double m_y,double r,double value);
        std::tuple<std::set<int>,double> get_empty_circ(double m_x, double m_y,double r,double value);
        std::tuple<std::set<int>,double> get_empty_tri(double x1, double y1,double x2, double y2,double x3,double y3,double value);


    private:
        int size_{};
        int m_size_{};
        double dx_{};
        double length_{};
        Eigen::MatrixXd L_{};
        Eigen::MatrixXd L_gen_{};
        Eigen::VectorXd b_{};
        Eigen::VectorXd x_b_{};
        std::vector<bool> boundary_{};
};





#endif