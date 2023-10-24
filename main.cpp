#include "corridor.h"
#include <time.h>
#include <iostream>
//#include <iomanip>
#include <sys/time.h>

#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <filesystem>




















int main(int argc, char **argv)
{
    srand( (unsigned)time( NULL ) );


    //
    struct timeval start, end;
    long long microseconds;
    gettimeofday(&start, nullptr);

    
    //int listNum = std::string(argv[1]);
    //td::vector<double> intersection_percnts(listNum);

    double xCoordinate = std::stod(argv[1]);
    double yCoordinate = std::stod(argv[2]);
    double zCoordinate = std::stod(argv[3]);
    double dx = std::stod(argv[4]);
    double dy = std::stod(argv[5]);
    double dz = std::stod(argv[6]);

    double intersection_percnt1 = std::stod(argv[7]);
    double intersection_percnt2 = std::stod(argv[8]);

    std::string file_path_1 = std::string(argv[9]);
    std::string file_path_2 = std::string(argv[10]);

    double tolerance = std::stod(argv[11]);
  
    std::vector<Mymesh> meshes;
    meshes.push_back(Mymesh(file_path_1));
    meshes.push_back(Mymesh(file_path_2));
    std::cout << "mesh number: " << meshes.size() << std::endl;
    for (Mymesh &mesh: meshes) mesh.create_aabb_tree();

    std::vector<double> intersection_percnts{intersection_percnt1, intersection_percnt2};
    Mytissue example_tissue(xCoordinate, yCoordinate, zCoordinate, dx, dy, dz);

    Surface_mesh corridor_mesh = create_corridor(meshes, example_tissue, intersection_percnts, tolerance);
    
    //
    gettimeofday(&end, nullptr);
    microseconds = (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
    // Print the elapsed time
    std::cout << "---------------main.cpp---------------Time elapsed: " << microseconds << " microseconds" << std::endl << std::endl << std::endl;

    
    std::string corridor_file_path = "corridor_pyramid_a.off";
    std::ofstream corridor_output(corridor_file_path);
    corridor_output << corridor_mesh;
 


    // // Directory path where your .off files are located
    // std::string directoryPath = std::string(argv[12]);//"D:/project/lcrcorridor/VH_F_Kidney_L";

    // // Create a vector to store the file paths
    // std::vector<std::string> fileAddresses;

    // // Iterate through the directory
    // for (const auto& entry : fs::directory_iterator(directoryPath)) {
    //     if (entry.path().extension() == ".off") {
    //         fileAddresses.push_back(entry.path().string());
    //     }
    // }

    // std::vector<Mymesh> meshList;

    // for (const std::string& str : fileAddresses) {
    //     std::cout << " " << str << std::endl;  // Dereference the iterator to get the string
    //     meshList.push_back(Mymesh(str));
    // }
    // std::cout << "mesh number: " << meshList.size() << std::endl;
    // for (Mymesh &mesh: meshList) mesh.create_aabb_tree();


    // OriginTissue origin_tissue(-0.139386, 0.0400516, 0.175776, 5, 6, 2);
    // origin_tissue.setTranslationDistanceXYZ(50.009, 65.904, 60.555);
    // origin_tissue.setRotationAngleXYZ(105, 131, 44);

    // std::cout << " do we get here"  << std::endl;

    // std::vector<double> testPercnt = compute_intersection_percnt_for_originTissue_meshes(meshList, origin_tissue);

    // for (const double& value : testPercnt) {
    //     std::cout << " " << value << std::endl;
    // }



}


