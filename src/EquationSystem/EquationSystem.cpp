#include "EquationSystem.hpp"



/////////////////////////////////////////////////////
// Determine Laplacian without boundary conditions //
/////////////////////////////////////////////////////
void EquationSystem::generate_init_L(){
    for(int k =0;k<m_size_;k++){
        int i = k%size_;
        int j = k/size_;
        L_gen_(k,k)=-4;
        if(i>0){
            L_gen_(k,i-1+j*size_)=1;
        }
        if(j>0){

            L_gen_(k,i+(j-1)*size_)=1;
        }
        if(i<size_-1){

            L_gen_(k,(i+1)+j*size_)=1;
        }
        if(j<size_-1){

            L_gen_(k,i+(j+1)*size_)=1;
        }
    }
    std::cout<< "Generated Laplace operator without boundary conditions\n";
}


//////////////////////////////////////////////////
// Solve equation system using Standard methods //
//////////////////////////////////////////////////
std::vector<double> EquationSystem::get_solution(std::string type_of_solver){
    std::cout<<"Solving equation system...\n";
    Eigen::VectorXd x_zw{};
    
   
    if(type_of_solver == "LUparallel"){
        x_zw = LinearSolver::solve_LU(L_,b_,true);
    }else if(type_of_solver == "LUsequential"){
        x_zw = LinearSolver::solve_LU(L_,b_,false);
    }else if(type_of_solver == "Jacobiparallel"){
        x_zw = LinearSolver::solve_Jacobian(L_,b_,0.0001,true);
    }else if(type_of_solver == "Jacobisequential"){
        x_zw = LinearSolver::solve_Jacobian(L_,b_,0.0001,false);
    }else{
        throw std::invalid_argument( "recieved invalid solver" );
    }

    //combine information from boundaries and equation solution
    std::vector<double> x(m_size_);
    int iter1=0;
    for(int i = 0;i<boundary_.size();i++){
        if(boundary_[i] ==false){
            x[i]=x_zw[iter1];
            iter1++;
        }else{
            x[i]=x_b_[i];
        }
    }
    std::cout<<"Equation solution calculated.\n";
    x_=x;
    return x;
}


//////////////////////////////
// Generate equation system //
//////////////////////////////
//Generates equation system in form of L*x = b
void EquationSystem::generate_equation_system(){

    //Define boundary objects and Charge distribution
    SystemObjects all_boundaries{};
    // all_boundaries.add(get_line(0.4,0.4,0.4,0.8,1));
    // all_boundaries.add(get_empty_circ(0.6,0.6,0.1,1));
    all_boundaries.add(get_empty_tri(0.2,0.2,0.5,0.2,0.5,0.7,1));
    
    SystemObjects all_charge_dist{};
    // all_charge_dist.add(get_filled_rect(0,0,1,1,100));
    all_charge_dist.add(get_filled_circ(0.6,0.6,0.1,100));
    
    //Evaluates elements of b corresponding to boundaries
    for(int i =0;i<all_boundaries.obj_value_list.size();i++){
        Eigen::VectorXd zw=Eigen::VectorXd::Constant(m_size_,0);
        for(auto v:all_boundaries.obj_pos_list[i]){
            zw(v) =  all_boundaries.obj_value_list[i];
            x_b_[v] += all_boundaries.obj_value_list[i];
            boundary_[v]=true;
        }
        b_ =b_ - L_gen_*zw;
    }

    //Evaluates elements of b corresponding to charge distribution
    for(int i =0;i<all_charge_dist.obj_value_list.size();i++){
        Eigen::VectorXd zw=Eigen::VectorXd::Constant(m_size_,0);
        for(auto v:all_charge_dist.obj_pos_list[i]){
            zw(v) =  all_charge_dist.obj_value_list[i];
        }
        b_ =b_ - zw*dx_*dx_;
    }

    //Union all boundary coordinates sets
    std::set<int> to_remove={};
    for(int i = 0; i< all_boundaries.obj_pos_list.size();i++){
        std::set<int> zw;
        std::set_union(to_remove.begin(), to_remove.end(),
                all_boundaries.obj_pos_list[i].begin(), all_boundaries.obj_pos_list[i].end(),
                std::inserter(zw, zw.begin()));
        to_remove = zw;
    }
    
    //exclude all elements from x corresponding to boundary
    std::set<int> all_numbers;
    for(int j=0; j<m_size_; j++)
        all_numbers.insert(j);
    std::set<int> included{};
    std::set_difference(all_numbers.begin(), all_numbers.end(), to_remove.begin(), to_remove.end(),std::inserter(included, included.end()));
    
    //reshape L so that columns and rows corresponding to boundaries are neglected
    L_ = Eigen::MatrixXd::Zero(m_size_, included.size());
    std::set<int>::iterator iter = included.begin();
    for(int i =0; i<included.size();i++){
        L_.col(i)=L_gen_.col(*iter);
        iter++;
    }
    L_gen_ = L_;
    auto b_zw = b_;
    L_ = Eigen::MatrixXd::Zero(included.size(), included.size());
    b_ = Eigen::VectorXd::Zero(included.size());
    iter = included.begin();
    for(int i =0; i<included.size();i++){
        L_.row(i)=L_gen_.row(*iter);
        b_[i]=b_zw[*iter];
        iter++;
    }

}




//////////////////////////////////////////////////////
// Boundary/Charge distribution generator functions //
//////////////////////////////////////////////////////

//generates koordinates corresponding to line 
std::tuple<std::set<int>,double> EquationSystem::get_line(double x_start, double y_start,double x_end, double y_end,double value){
    std::set<int> crossing_pos;
    if(std::abs(x_start-x_end)*length_<dx_){
        int j_sign = (y_end > y_start) - (y_end < y_start);
        int i = int(x_start*length_/dx_);
        for(int j = int(y_start*length_/dx_);j_sign*j<j_sign*int(y_end*length_/dx_);j+=j_sign){
            crossing_pos.insert(i+j*size_);
        }
    }
    else if(std::abs(y_start-y_end)*length_<dx_){
        int i_sign = (x_end > x_start) - (x_end < x_start);
        int j = int(y_start*length_/dx_);
        for(int i = int(x_start*length_/dx_);i*i_sign< i_sign*int(x_end*length_/dx_);i+=i_sign){
            crossing_pos.insert(i+j*size_);
        }
    }else{
        x_start = length_*x_start;
        x_end = length_*x_end;
        y_start = length_*y_start;
        y_end = length_*y_end;

        int i_start = int(x_start/dx_),i_end = int(x_end/dx_);
        int j_start = int(y_start/dx_),j_end = int(y_end/dx_);

        double m = std::abs(double(j_end-j_start)/double(i_end-i_start));
        int i_sign = (i_end > i_start) - (i_end < i_start);
        int j_sign = (j_end > j_start) - (j_end < j_start);
        int i =0, j= 0;
        while(i<=std::abs(i_end-i_start) || j<=std::abs(j_end-j_start)){
            crossing_pos.insert((i_start+i_sign*i)+(j_start+j_sign*j)*size_);
            if(std::abs(double (j)/double(i))<=m){
                j++;
            }else{
                i++;
            }
        }
    }
    
    return std::make_tuple(crossing_pos,value);
}


//generates coordinates corresponding to filled rectangle(x1<x2,y1<y2)
std::tuple<std::set<int>,double> EquationSystem::get_filled_rect(double x1, double y1,double x2, double y2,double value){
    std::set<int> crossing_pos;
    int zw = 0;
    if(x1 > x2){
        zw = x1;
        x1 = x2;
        x2 = zw;
    }
    if(y1 > y2){
        zw = y1;
        y1 = y2;
        y2 = zw;
    }
    for(int k =0;k<m_size_;k++){
        int i = k%size_;
        int j = k/size_;  
        if(i*dx_>=x1*length_ && i*dx_<=x2*length_){
            if(j*dx_>=y1*length_ && j*dx_<=y2*length_){
                crossing_pos.insert(k);
            }
        } 
    }
    return std::make_tuple(crossing_pos,value);
}

//generates koordinates corresponding to rectangle without filling
std::tuple<std::set<int>,double> EquationSystem::get_empty_rect(double x1, double y1,double x2, double y2,double value){
    std::vector<std::set<int>> boundaries(4);
    boundaries[0] = std::get<0>(get_line(x1,y1,x1,y2,value));
    boundaries[1] = std::get<0>(get_line(x2,y1,x2,y2,value));
    boundaries[2] = std::get<0>(get_line(x1,y1,x2,y1,value));
    boundaries[3] = std::get<0>(get_line(x1,y2,x2,y2,value));
    std::set<int> result={};
    for(int i = 0; i< boundaries.size();i++){
        std::set<int> zw;
        std::set_union(result.begin(), result.end(),
                boundaries[i].begin(), boundaries[i].end(),
                std::inserter(zw, zw.begin()));
        result = zw;
    }
    return std::make_tuple(result,value);
}

//generates koordinates corresponding to filled circle
std::tuple<std::set<int>,double> EquationSystem::get_filled_circ(double m_x, double m_y,double r,double value){
    std::set<int> crossing_pos;
    for(int k =0;k<m_size_;k++){
        int i = k%size_;
        int j = k/size_;  
        if(std::pow(i*dx_-m_x*length_,2)+std::pow(j*dx_-m_y*length_,2)<=r*length_*r*length_){
            crossing_pos.insert(k);
        } 
    }
    return std::make_tuple(crossing_pos,value);
}

//generates koordinates corresponding to circle without filling
std::tuple<std::set<int>,double> EquationSystem::get_empty_circ(double m_x, double m_y,double r,double value){
    std::set<int> crossing_pos;
    for(int k =0;k<m_size_;k++){
        int i = k%size_;
        int j = k/size_;  
        if(std::pow(i*dx_-m_x*length_,2)+std::pow(j*dx_-m_y*length_,2)<=std::pow(length_*r+std::sqrt(2)*dx_,2) && std::pow(i*dx_-m_x*length_,2)+std::pow(j*dx_-m_y*length_,2)>=r*length_*r*length_){
            crossing_pos.insert(k);
        } 
    }
    return std::make_tuple(crossing_pos,value);
}

//generates coordinates corresponding to triangle without filling
std::tuple<std::set<int>,double> EquationSystem::get_empty_tri(double x1, double y1,double x2, double y2,double x3,double y3,double value){
    std::vector<std::set<int>> boundaries(3);
    boundaries[0] = std::get<0>(get_line(x1,y1,x2,y2,value));
    boundaries[1] = std::get<0>(get_line(x2,y2,x3,y3,value));
    boundaries[2] = std::get<0>(get_line(x3,y3,x1,y1,value));
    std::set<int> result={};
    for(int i = 0; i< boundaries.size();i++){
        std::set<int> zw;
        std::set_union(result.begin(), result.end(),
                boundaries[i].begin(), boundaries[i].end(),
                std::inserter(zw, zw.begin()));
        result = zw;
    }
    
    return std::make_tuple(result,value);
}


//////////////////////
// Output functions //           
//////////////////////
void EquationSystem::write_to_csv(){
    if(x_.empty()){
        std::cout<<"Solution has not yet been calculated!!!\n";
        return;
    }
    std::ofstream file;
    file.open("../Data/system.csv");
    file<<"x,y,value"<<"\n";
    for(int k=0;k<x_.size();k++){
        int i = k%size_;
        int j = k/size_;
        file<<i*dx_<<","<<dx_*j<<","<<x_[k]<<"\n";
    }
    file.close();
    std::cout<<"Finished writing to csv file.\n";
}

void EquationSystem::write_to_vector(std::vector<double> &x_data,std::vector<double> &y_data,std::vector<double> &z_data){
    if(x_.empty()){
        std::cout<<"Solution has not yet been calculated!!!\n";
        return;
    }
    for(int k=0;k<x_.size();k++){
        int i = k%size_;
        int j = k/size_;
        x_data.push_back(i*dx_);
        y_data.push_back(j*dx_);
        z_data.push_back(x_[k]);
    }
}

