#pragma once

#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Real_timer.h>

#include "algo.h"
#include "utils.h"
#include <cmath>

class Mytissue: public Mymesh {

    public:
        Mytissue() = default;
        Mytissue(double c_x, double c_y, double c_z, double d_x, double d_y, double d_z); 
        
        double center_x, center_y, center_z;
        double dimension_x, dimension_y, dimension_z;
    
    public:
        std::vector<Point> &get_points();
    
    private:
        std::vector<Point> &generate_points(int resolution);
        // Surface_mesh create_mesh();

    private:
        std::vector<Point> points;


};


class OriginTissue: public Mytissue {

    public:
        OriginTissue() = default;
        OriginTissue(double c_x, double c_y, double c_z, double d_x, double d_y, double d_z); 
        
        double center_x, center_y, center_z;
        double dimension_x, dimension_y, dimension_z;
    
    public:
        std::vector<Point> &get_points();
        void setTranslationDistanceXYZ(double distance_x, double distance_y, double distance_z);
        void setRotationAngleXYZ(double angle_x, double angle_y, double angle_z);
    
    private:
        std::vector<Point> &generate_points(int resolution);
        void applyTranslation(double& x, double& y, double& z, double dx, double dy, double dz);
        void applyEulerRotations(double& x, double& y, double& z, double angleX, double angleY, double angleZ);
        //set how many points we split the tissue block into
        //void setNumOfPoints(int x, int y, int z);

        std::vector<Point> points;
        //translation distance along x, y and z axes
        double translate_x = 0;
        double translate_y = 0;
        double translate_z = 0;
        // Euler rotations angles around x, y and z axes
        double alpha = 0; 
        double beta = 0; 
        double gamma = 0;
        int numOfPoints = 1;


};


std::vector<Point> find_all_locations(Mymesh &my_mesh, Mytissue &example_tissue, double intersection_percentage, double tolerance);

Surface_mesh create_corridor(std::vector<Mymesh> &meshes, Mytissue &example_tissue, std::vector<double> &intersection_percnts, double tolerance);

Surface_mesh create_corridor(std::vector<Mymesh> &organ, Mytissue &example_tissue, std::vector<std::pair<int, double>> &result, double tolerance);

std::vector<Point> create_point_cloud_corridor_for_multiple_AS(std::vector<Mymesh> &meshes, Mytissue &example_tissue, std::vector<double> &intersection_percnts, double tolerance);

std::vector<Point> create_point_cloud_corridor_for_multiple_AS(std::vector<Mymesh> &organ, Mytissue &example_tissue, std::vector<std::pair<int, double>> &result, double tolerance);

Surface_mesh create_corridor_from_point_cloud(std::vector<Point> &points);

double compute_intersection_volume(Mymesh &AS, Mytissue &tissue);
//double compute_intersection_volume_serial(Mymesh &AS, Mytissue &tissue);

double generate_pertubation(double step);




std::vector<double> compute_intersection_percnt_for_originTissue_meshes(std::vector<Mymesh> &meshes, OriginTissue &originTissueBlock);
