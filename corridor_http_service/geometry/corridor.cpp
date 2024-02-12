#include "corridor.h"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <sys/time.h>
using namespace std;
   

Mytissue::Mytissue(double c_x, double c_y, double c_z, double d_x, double d_y, double d_z)
:Mymesh(c_x, c_y, c_z, d_x, d_y, d_z),
center_x(c_x), center_y(c_y), center_z(c_z), dimension_x(d_x), dimension_y(d_y), dimension_z(d_z)
{

    create_aabb_tree();
    generate_points(10);

}


std::vector<Point> &Mytissue::get_points()
{
    return this->points;
}


std::vector<Point> &Mytissue::generate_points(int resolution=10)
{

    double min_x = center_x - dimension_x/2, min_y = center_y - dimension_y/2, min_z = center_z - dimension_z/2;
    double max_x = center_x + dimension_x/2, max_y = center_y + dimension_y/2, max_z = center_z + dimension_z/2; 
    double delta_x = (max_x - min_x) / resolution, delta_y = (max_y - min_y) / resolution, delta_z = (max_z - min_z) / resolution;    
  

    for (int i = 0; i < resolution; i++)
        for (int j = 0; j < resolution; j++)
            for (int k = 0; k < resolution; k++)
            {
                double c_x = min_x + (i + 0.5) * delta_x;
                double c_y = min_y + (j + 0.5) * delta_y;
                double c_z = min_z + (k + 0.5) * delta_z;
                
                Point p(c_x, c_y, c_z);
                points.push_back(p); 
            }
    
    return points;
}




// //original tissue block obtained from json file
// OriginTissue::OriginTissue(double c_x, double c_y, double c_z, double d_x, double d_y, double d_z)
// :Mytissue(c_x, c_y, c_z, d_x, d_y, d_z),
// origin_x(c_x), origin_y(c_y), origin_z(c_z), dimension_x(d_x), dimension_y(d_y), dimension_z(d_z)
// {

//     create_aabb_tree();
//     generate_points(10);

// }

// std::vector<Point> &OriginTissue::get_points()
// {
//     return this->points;
// }

// //apply translation along x, y and z axes
// // void OriginTissue::applyTranslation(double& x, double& y, double& z, double dx, double dy, double dz)
// // {
// //     x = x + dx;
// //     y = y + dy;
// //     z = z + dz;
// // }

// // //apply Euler rotation around x, y and z axes
// // void OriginTissue::applyEulerRotations(double& x, double& y, double& z, double angleX, double angleY, double angleZ) 
// // {

// //     double radianX = angleX * (M_PI / 180.0);
// //     double radianY = angleY * (M_PI / 180.0);
// //     double radianZ = angleZ * (M_PI / 180.0);


// //     // Apply x-rotation
// //     double tempY = y;
// //     y = cos(radianX) * y - sin(radianX) * z;
// //     z = sin(radianX) * tempY + cos(radianX) * z;

// //     // Apply y-rotation
// //     double tempX = x;
// //     x = cos(radianY) * x + sin(radianY) * z;
// //     z = -sin(radianY) * tempX + cos(radianY) * z;

// //     // Apply z-rotation
// //     tempX = x;
// //     x = cos(radianZ) * x - sin(radianZ) * y;
// //     y = sin(radianZ) * tempX + cos(radianZ) * y;

// // }

// // void OriginTissue::setNumOfPoints(int x, int y, int z)
// // {
// //     numOfPoints = x * y * z;
// // }

// std::vector<Point> &OriginTissue::generate_points(int resolution=10)
// {   
//     //setNumOfPoints(resolution, resolution, resolution);
//     std::cout << " OriginTissue::generate_points " << std::endl;

//     double min_x = -dimension_x/2, min_y = -dimension_y/2, min_z = -dimension_z/2;
//     double max_x = dimension_x/2, max_y = dimension_y/2, max_z = dimension_z/2; 
//     double delta_x = (max_x - min_x) / resolution, delta_y = (max_y - min_y) / resolution, delta_z = (max_z - min_z) / resolution;    
  
//     Eigen::Vector3d origin(origin_x, origin_y, origin_z);

//     Eigen::MatrixXd R_x(3, 3);
//     Eigen::MatrixXd R_y(3, 3);
//     Eigen::MatrixXd R_z(3, 3);

//     double radius_x = x_rotation * PI / 180;
//     double radius_y = y_rotation * PI / 180;
//     double radius_z = z_rotation * PI / 180;
//     R_x << 1.0, 0.0, 0.0, 
//         0.0, cos(radius_x), -sin(radius_x),
//         0.0, sin(radius_x), cos(radius_x);

//     R_y << cos(radius_y), 0.0, sin(radius_y),
//         0.0, 1.0, 0.0,
//         -sin(radius_y), 0.0, cos(radius_y);

//     R_z << cos(radius_z), -sin(radius_z), 0.0, 
//         sin(radius_z), cos(radius_z), 0.0,
//         0.0, 0.0, 1.0;

//     Eigen::MatrixXd rotationMatrix = R_x * R_y * R_z;
//     Eigen::Vector3d translation(translation_x, translation_y, translation_z);

//     double center_x, center_y, center_z;
//     for (int i = 0; i < resolution; i++)
//         for (int j = 0; j < resolution; j++)
//             for (int k = 0; k < resolution; k++)
//             {
//                 center_x = min_x + (i + 0.5) * delta_x;
//                 center_y = min_y + (j + 0.5) * delta_y;
//                 center_z = min_z + (k + 0.5) * delta_z;
                
//                 //apply translation along x, y and z axes
//                 //applyTranslation(c_x, c_y, c_z, translate_x, translate_y, translate_z);
//                 //apply Euler rotation around x, y and z axes
//                 //applyEulerRotations(c_x, c_y, c_z, alpha, beta, gamma);    

//                 Eigen::Vector3d temp_vec(center_x, center_y, center_z);
//                 temp_vec = (rotationMatrix * temp_vec + T)/1000.0 + origin;

//                 Point p(temp_vec(0), temp_vec(1), temp_vec(2));
//                 points.push_back(p); 
//             }
    
//     return points;
// }

// // void OriginTissue::setTranslationDistanceXYZ(double distance_x, double distance_y, double distance_z)
// // {
// //     translate_x = distance_x;
// //     translate_y = distance_y;
// //     translate_z = distance_z;
// // }

// // void OriginTissue::setRotationAngleXYZ(double angle_x, double angle_y, double angle_z)
// // {
// //     alpha = angle_x; 
// //     beta = angle_y; 
// //     gamma = angle_z;
// // }







// //computer intersection percentage between original tissue block and a list of meshes.
// std::vector<double> compute_intersection_percnt_for_originTissue_meshes(std::vector<Mymesh> &meshes, OriginTissue &originTissueBlock) 
// {

//     double tissue_d_x = originTissueBlock.dimension_x;
//     double tissue_d_y = originTissueBlock.dimension_y;
//     double tissue_d_z = originTissueBlock.dimension_z;

//     //calculate tissue block volume
//     //double tbv = tissue_d_x * tissue_d_y * tissue_d_z;

//     //find MBB for tissue block
//     CGAL::Bbox_3 tissueBBox = PMP::bbox(originTissueBlock.get_raw_mesh());

//     //std::cout << " test 1 "  << std::endl;

//     std::vector<double> intersect_percnt;

//     //calculate if the bbox of mesh intersects the bbox of tissue for each mesh.
//     for (int i = 0; i < meshes.size(); i++) {

//         std::cout << i << std::endl << std::endl;

//         //std::cout << " test 2 "  << std::endl;

//         CGAL::Bbox_3 meshBBox = PMP::bbox(meshes[i].get_raw_mesh());

//         //std::cout << " test 3 "  << std::endl;

//         double lcr1 = meshBBox.xmin();
//         double lcr2 = tissueBBox.xmax();
//         double lcr3 = meshBBox.xmax();
//         double lcr4 = tissueBBox.xmin();

//         double lcr11 = meshBBox.ymin();
//         double lcr21 = tissueBBox.ymax();
//         double lcr31 = meshBBox.ymax();
//         double lcr41 = tissueBBox.ymin();

//         double lcr111 = meshBBox.zmin();
//         double lcr211 = tissueBBox.zmax();
//         double lcr311 = meshBBox.zmax();
//         double lcr411 = tissueBBox.zmin();



//         //std::cout << " test 4 "  << std::endl;

        

//         bool intersectX = (meshBBox.xmin() <= tissueBBox.xmax()) && (meshBBox.xmax() >= tissueBBox.xmin());
//         bool intersectY = (meshBBox.ymin() <= tissueBBox.ymax()) && (meshBBox.ymax() >= tissueBBox.ymin());
//         bool intersectZ = (meshBBox.zmin() <= tissueBBox.zmax()) && (meshBBox.zmax() >= tissueBBox.zmin());
//         //std::cout << " test 5 "  << std::endl;
//     //     // If there is an intersection along all three axes, the bounding boxes intersect
//         if (intersectX && intersectY && intersectZ) {
            
//             //calculate intersection percentages
//             //cut tissue block into 1000 small boxes, use center point to represent each small box,
//             //for each center point, check if it is inside certain mesh,
//             //intersection percentage = Number of points inside / total number of points
//             std::cout << " BBX intersect "  << std::endl;
//             double intersection_volume = compute_intersection_volume(meshes[i], originTissueBlock);

//             std::cout << " intersection_volume: " << intersection_volume << std::endl;
//             //intersect_percnt[i] = intersection_volume / (tissue_d_x * tissue_d_y * tissue_d_z);
//             //std::cout << " test 6 "  << std::endl;
//             intersect_percnt.push_back(intersection_volume / (tissue_d_x * tissue_d_y * tissue_d_z));
//             //std::cout << " test 7 "  << std::endl;

//         } else {    //BBX do not intersect
//             std::cout << " BBX do not intersect "  << std::endl;
//             intersect_percnt.push_back(0);
//             //std::cout << " test 9 "  << std::endl;
//         }
        
//     }

//     return intersect_percnt;
// }







 








std::vector<Point> find_all_locations(Mymesh &my_mesh, Mytissue &example_tissue, double intersection_percentage, double tolerance)
{

    std::vector<Point> center_path;

    // double example_intersection_volume = compute_intersection_volume(my_mesh, example_tissue);
    double example_intersection_volume = example_tissue.dimension_x * example_tissue.dimension_y * example_tissue.dimension_z * intersection_percentage;

    double step_x = example_tissue.dimension_x / 5;
    double step_y = example_tissue.dimension_y / 5;
    double step_z = example_tissue.dimension_z / 5;

    double example_d_x = example_tissue.dimension_x;
    double example_d_y = example_tissue.dimension_y;
    double example_d_z = example_tissue.dimension_z;

    Surface_mesh mesh = my_mesh.get_raw_mesh();
    CGAL::Bbox_3 bbox = PMP::bbox(mesh);

    double x_min = bbox.xmin();
    double y_min = bbox.ymin();
    double z_min = bbox.zmin();
    double x_max = bbox.xmax();
    double y_max = bbox.ymax();
    double z_max = bbox.zmax();

    for (double c_x = x_min - example_d_x / 2; c_x < x_max + example_d_x / 2; c_x += step_x)
        for (double c_y = y_min - example_d_y / 2; c_y < y_max + example_d_y / 2; c_y += step_y)
            for (double c_z = z_min - example_d_z / 2; c_z < z_max + example_d_z / 2; c_z += step_z)
            {
                Mytissue cur_tissue(c_x, c_y, c_z, example_d_x, example_d_y, example_d_z);
                double intersection_volume = compute_intersection_volume(my_mesh, cur_tissue);
                
                if (std::abs(intersection_volume - example_intersection_volume) <= tolerance * example_intersection_volume)
                    center_path.push_back(Point(c_x, c_y, c_z));

            }
    
    return center_path;
    

}

Surface_mesh create_corridor(std::vector<Mymesh> &meshes, Mytissue &example_tissue, std::vector<double> &intersection_percnts, double tolerance)
{

    std::vector<Point> point_cloud = create_point_cloud_corridor_for_multiple_AS(meshes, example_tissue, intersection_percnts, tolerance);
    return create_corridor_from_point_cloud(point_cloud);

}

// overload: create corridor based on the collision detection result
Surface_mesh create_corridor(std::vector<Mymesh> &organ, Mytissue &example_tissue, std::vector<std::pair<int, double>> &result, double tolerance)
{
    std::vector<Point> point_cloud = create_point_cloud_corridor_for_multiple_AS(organ, example_tissue, result, tolerance);
    return create_corridor_from_point_cloud(point_cloud);

}

double generate_pertubation(double step)
{
    return step * ((double)rand()/RAND_MAX * 2.0 - 1.0);
}

std::vector<Point> create_point_cloud_corridor_for_multiple_AS(std::vector<Mymesh> &meshes, Mytissue &example_tissue, std::vector<double> &intersection_percnts, double tolerance)
{

    std::vector<Point> center_path;
    std::vector<Point> point_cloud;

    double intersect_x_min = -1e10, intersect_y_min = -1e10, intersect_z_min = -1e10;
    double intersect_x_max = 1e10, intersect_y_max = 1e10, intersect_z_max = 1e10;  

    // double step_x = example_tissue.dimension_x / 4.9;
    // double step_y = example_tissue.dimension_y / 4.9;
    // double step_z = example_tissue.dimension_z / 4.9;

    double example_d_x = example_tissue.dimension_x;
    double example_d_y = example_tissue.dimension_y;
    double example_d_z = example_tissue.dimension_z;

    double tbv = example_d_x * example_d_y * example_d_z;

    for (Mymesh &mesh: meshes)
    {
        CGAL::Bbox_3 bbox = PMP::bbox(mesh.get_raw_mesh());
        intersect_x_min = std::max(intersect_x_min, bbox.xmin());
        intersect_y_min = std::max(intersect_y_min, bbox.ymin());
        intersect_z_min = std::max(intersect_z_min, bbox.zmin());

        intersect_x_max = std::min(intersect_x_max, bbox.xmax());
        intersect_y_max = std::min(intersect_y_max, bbox.ymax());
        intersect_z_max = std::min(intersect_z_max, bbox.zmax());

    }

    double step_x = (intersect_x_max - intersect_x_min + example_d_x) / 40.0;
    double step_y = (intersect_y_max - intersect_y_min + example_d_y) / 40.0;
    double step_z = (intersect_z_max - intersect_z_min + example_d_z) / 40.0;

    std::cout << "min x, y, z: " << intersect_x_min << " " << intersect_y_min << " " << intersect_z_min << std::endl;
    std::cout << "max x, y, z: " << intersect_x_max << " " << intersect_y_max << " " << intersect_z_max << std::endl;
    std::cout << "step size: " << step_x << " " << step_y << " " << step_z << std::endl;
    
    //count the number of loop of x, y, z axis, since OPENMP does not work with double loop
    int x_loop, y_loop, z_loop;
    x_loop = ceil(((intersect_x_max + example_d_x / 2) - (intersect_x_min - example_d_x / 2)) / step_x);
    y_loop = ceil(((intersect_y_max + example_d_y / 2) - (intersect_y_min - example_d_y / 2)) / step_y);
    z_loop = ceil(((intersect_z_max + example_d_z / 2) - (intersect_z_min - example_d_z / 2)) / step_z);
    cout << " X_loop: " << x_loop << "-----Y_loop: " << y_loop << "-----Z_loop: " << y_loop << std::endl;
                    
    struct timeval start, end;
    long long microseconds;
    gettimeofday(&start, nullptr);

    #pragma omp parallel for schedule(dynamic)
    //double c_x, c_y, c_z;
    for //(double c_x = intersect_x_min - example_d_x / 2; c_x < intersect_x_max + example_d_x / 2; c_x += step_x)
        (int x_count = 0; x_count < x_loop; x_count += 1)
        for //(double c_y = intersect_y_min - example_d_y / 2; c_y < intersect_y_max + example_d_y / 2; c_y += step_y)
            (int y_count = 0; y_count < y_loop; y_count += 1)
            for//(double c_z = intersect_z_min - example_d_z / 2; c_z < intersect_z_max + example_d_z / 2; c_z += step_z)
                (int z_count = 0; z_count < z_loop; z_count += 1)
            {
                // std::cout << c_x << " " << c_y << " " << c_z << std::endl;
                double c_x, c_y, c_z;
                c_x = intersect_x_min - example_d_x / 2 + (step_x * x_count);
                c_y = intersect_y_min - example_d_y / 2 + (step_y * y_count); 
                c_z = intersect_z_min - example_d_z / 2 + (step_z * z_count);
                Mytissue cur_tissue(c_x, c_y, c_z, example_d_x, example_d_y, example_d_z);
                
                bool is_in_corridor = true;
                for (int i = 0; i < meshes.size(); i++)
                {
                    Mymesh &mesh = meshes[i];
                    double example_intersection_volume = tbv * intersection_percnts[i];

                    double intersection_volume = compute_intersection_volume(mesh, cur_tissue);
                    // std::cout << i << " example intersection volume: " << example_intersection_volume << " " << intersection_volume << std::endl;
                    if (std::abs(intersection_volume - example_intersection_volume) > tolerance * example_intersection_volume)
                    {
                        is_in_corridor = false;
                        break;
                    }

                }

                if (is_in_corridor)
                {
                    center_path.push_back(Point(c_x, c_y, c_z));
                    //std::cout << generate_pertubation(step_x) << " " << generate_pertubation(step_y) << " " << generate_pertubation(step_z) << std::endl;
                    point_cloud.push_back(Point(c_x - example_d_x / 2 , c_y - example_d_y / 2 , c_z - example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x + example_d_x / 2 , c_y - example_d_y / 2 , c_z - example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x - example_d_x / 2 , c_y + example_d_y / 2 , c_z - example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x + example_d_x / 2 , c_y + example_d_y / 2 , c_z - example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x - example_d_x / 2 , c_y - example_d_y / 2 , c_z + example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x + example_d_x / 2 , c_y - example_d_y / 2 , c_z + example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x - example_d_x / 2 , c_y + example_d_y / 2 , c_z + example_d_z / 2 ));
                    point_cloud.push_back(Point(c_x + example_d_x / 2 , c_y + example_d_y / 2 , c_z + example_d_z / 2 ));

                }

            }

    gettimeofday(&end, nullptr);
    microseconds = (end.tv_sec - start.tv_sec) * 1000000LL + (end.tv_usec - start.tv_usec);
    // Print the elapsed time
    std::cout << "---Multithreaded---corridor-generation---Time elapsed: " << microseconds << " microseconds" << std::endl;        
    return point_cloud;

}

//overload: create point cloud based on the collision detection result
std::vector<Point> create_point_cloud_corridor_for_multiple_AS(std::vector<Mymesh> &organ, Mytissue &example_tissue, std::vector<std::pair<int, double>> &result, double tolerance)
{

    std::vector<Point> center_path;
    std::vector<Point> point_cloud;

    double intersect_x_min = -1e10, intersect_y_min = -1e10, intersect_z_min = -1e10;
    double intersect_x_max = 1e10, intersect_y_max = 1e10, intersect_z_max = 1e10;  

    // double step_x = example_tissue.dimension_x / 4.9;
    // double step_y = example_tissue.dimension_y / 4.9;
    // double step_z = example_tissue.dimension_z / 4.9;

    double example_d_x = example_tissue.dimension_x;
    double example_d_y = example_tissue.dimension_y;
    double example_d_z = example_tissue.dimension_z;

    double tbv = example_d_x * example_d_y * example_d_z;

    for (auto s: result)
    {
        Mymesh &mesh = organ[s.first];
        CGAL::Bbox_3 bbox = PMP::bbox(mesh.get_raw_mesh());
        intersect_x_min = std::max(intersect_x_min, bbox.xmin());
        intersect_y_min = std::max(intersect_y_min, bbox.ymin());
        intersect_z_min = std::max(intersect_z_min, bbox.zmin());

        intersect_x_max = std::min(intersect_x_max, bbox.xmax());
        intersect_y_max = std::min(intersect_y_max, bbox.ymax());
        intersect_z_max = std::min(intersect_z_max, bbox.zmax());
    }

    double step_x = (intersect_x_max - intersect_x_min + example_d_x) / 40.0;
    double step_y = (intersect_y_max - intersect_y_min + example_d_y) / 40.0;
    double step_z = (intersect_z_max - intersect_z_min + example_d_z) / 40.0;
    
    std::cout << "min x, y, z: " << intersect_x_min << " " << intersect_y_min << " " << intersect_z_min << std::endl;
    std::cout << "max x, y, z: " << intersect_x_max << " " << intersect_y_max << " " << intersect_z_max << std::endl;
    std::cout << "step size: " << step_x << " " << step_y << " " << step_z << std::endl;

    
    for (double c_x = intersect_x_min - example_d_x / 2; c_x < intersect_x_max + example_d_x / 2; c_x += step_x)
        for (double c_y = intersect_y_min - example_d_y / 2; c_y < intersect_y_max + example_d_y / 2; c_y += step_y)
            for (double c_z = intersect_z_min - example_d_z / 2; c_z < intersect_z_max + example_d_z / 2; c_z += step_z)
            {
                // std::cout << c_x << " " << c_y << " " << c_z << std::endl;
                Mytissue cur_tissue(c_x, c_y, c_z, example_d_x, example_d_y, example_d_z);
                
                bool is_in_corridor = true;
                for (auto s: result)
                {
                    Mymesh &mesh = organ[s.first];
                    double example_intersection_volume = tbv * s.second;
                    double intersection_volume = compute_intersection_volume(mesh, cur_tissue);
                    if (std::abs(intersection_volume - example_intersection_volume) > tolerance * example_intersection_volume)
                    {
                        is_in_corridor = false;
                        break;
                    }

                }

                if (is_in_corridor)
                {
                    center_path.push_back(Point(c_x, c_y, c_z));
                    point_cloud.push_back(Point(c_x - example_d_x / 2, c_y - example_d_y / 2, c_z - example_d_z / 2));
                    point_cloud.push_back(Point(c_x + example_d_x / 2, c_y - example_d_y / 2, c_z - example_d_z / 2));
                    point_cloud.push_back(Point(c_x - example_d_x / 2, c_y + example_d_y / 2, c_z - example_d_z / 2));
                    point_cloud.push_back(Point(c_x + example_d_x / 2, c_y + example_d_y / 2, c_z - example_d_z / 2));
                    point_cloud.push_back(Point(c_x - example_d_x / 2, c_y - example_d_y / 2, c_z + example_d_z / 2));
                    point_cloud.push_back(Point(c_x + example_d_x / 2, c_y - example_d_y / 2, c_z + example_d_z / 2));
                    point_cloud.push_back(Point(c_x - example_d_x / 2, c_y + example_d_y / 2, c_z + example_d_z / 2));
                    point_cloud.push_back(Point(c_x + example_d_x / 2, c_y + example_d_y / 2, c_z + example_d_z / 2));

                }

            }


    return point_cloud;

}


Surface_mesh create_corridor_from_point_cloud(std::vector<Point> &points)
{
    Surface_mesh wrap;
    
    std::cout << "points size: " << points.size() << std::endl;
    if (points.size() == 0)
        return wrap;

    // Compute the alpha and offset values
    const double relative_alpha = 10.;
    const double relative_offset = 300.;
    
    CGAL::Bbox_3 bbox = CGAL::bbox_3(std::cbegin(points), std::cend(points));
    const double diag_length = std::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                                        CGAL::square(bbox.ymax() - bbox.ymin()) +
                                        CGAL::square(bbox.zmax() - bbox.zmin()));

    const double alpha = diag_length / relative_alpha;
    const double offset = diag_length / relative_offset;
    std::cout << "absolute alpha = " << alpha << " absolute offset = " << offset << std::endl;
    // Construct the wrap
    CGAL::Real_timer t;
    
    t.start();
    std::cout << "size of point cloud: " << points.size() << std::endl;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    t.stop();

    return wrap;

}


// compute intersection volume between a mesh and an axis-aligned tissue 
double compute_intersection_volume(Mymesh &AS, Mytissue &tissue)
{

    auto aabbtree_AS = AS.get_aabb_tree();
    auto aabbtree_tissue = tissue.get_aabb_tree();

    double percentage = 0.0;
    if (aabbtree_AS->do_intersect(*aabbtree_tissue))
    {
        percentage = AS.percentage_points_inside(tissue.get_points());
    }
    else
    {
        Surface_mesh &tissue_raw_mesh = tissue.get_raw_mesh();
        
        // the tissue block is wholely inside the anatomical structure. 
        bool is_contain_1 = true;
        for (auto vd: tissue_raw_mesh.vertices())
        {
            Point p = tissue_raw_mesh.point(vd);

            if (!AS.point_inside(p)) is_contain_1 = false;
            break;

        }


        // the anatomical structure is wholely inside the tissue block, still use the voxel-based algorithm, can be simplified to use the volume of the anatomical structure. 
        bool is_contain_2 = true;
        Surface_mesh &AS_raw_mesh = AS.get_raw_mesh();

        for (auto vd: AS_raw_mesh.vertices())
        {
            Point p = AS_raw_mesh.point(vd);

            if (!tissue.point_inside(p)) is_contain_2 = false;
            break;
        
        }

        if (is_contain_1) percentage = 1.0;
        else if (is_contain_2) percentage = AS.percentage_points_inside(tissue.get_points());
    }
    //std::cout << " tissue.dimension_xyz : " << tissue.dimension_x << "---" << tissue.dimension_y << "---" << tissue.dimension_z << std::endl;
    //std::cout << " percentage : " << percentage << std::endl;
    double volume = percentage * tissue.dimension_x * tissue.dimension_y * tissue.dimension_z;
    
    return volume;
    
}

/*
double compute_intersection_volume_serial(Mymesh &AS, Mytissue &tissue)
{

    auto aabbtree_AS = AS.get_aabb_tree();
    auto aabbtree_tissue = tissue.get_aabb_tree();

    double percentage = 0.0;
    if (aabbtree_AS->do_intersect(*aabbtree_tissue))
    {
        //run 1000 times
        percentage = AS.percentage_points_inside_serial(tissue.get_points());
    }
    else
    {
        Surface_mesh &tissue_raw_mesh = tissue.get_raw_mesh();       
        // the tissue block is wholely inside the anatomical structure. 
        bool is_contain_1 = true;
        // break for loop
        for (auto vd: tissue_raw_mesh.vertices())
        {
            Point p = tissue_raw_mesh.point(vd);
            if (!AS.point_inside(p)) is_contain_1 = false;
            break;
        }

        // the anatomical structure is wholely inside the tissue block, still use the voxel-based algorithm, can be simplified to use the volume of the anatomical structure. 
        bool is_contain_2 = true;
        Surface_mesh &AS_raw_mesh = AS.get_raw_mesh();

        for (auto vd: AS_raw_mesh.vertices())
        {
            Point p = AS_raw_mesh.point(vd);

            if (!tissue.point_inside(p)) is_contain_2 = false;
            break;        
        }
        if (is_contain_1) percentage = 1.0;
        else if (is_contain_2) percentage = AS.percentage_points_inside_serial(tissue.get_points());
    }
    double volume = percentage * tissue.dimension_x * tissue.dimension_y * tissue.dimension_z;
    
    return volume;
    
}*/