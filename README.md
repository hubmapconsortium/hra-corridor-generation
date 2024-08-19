# 3D Corridor for Tissue Block Re-registration

**Version::** 1.0.0

**Release date:** 9 Feb 2024

## Overview:
A C++ library for http service for 3D corridor generation given a RUI Registration.


## Dependencies:
For C++ libraries:

1. [Download CGAL 5.5.3](https://github.com/CGAL/cgal/releases/download/v5.5.3/CGAL-5.5.3.zip)
    Extract the compressed file to the 'corridor_http_service' folder.

2. CMake
    ```bash
    sudo apt-get install build-essential libssl-dev
    cd /tmp
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    ./bootstrap
    make
    sudo make install
    ```
3. Boost
    ```bash
    sudo apt-get update
    sudo apt-get install libboost-all-dev
    ```
4. GMP
    ```bash
    sudo apt-get install libgmp-dev
    ```
5. MPFR
    ```bash
    sudo apt-get install libmpfr-dev
    ```
6. Eigen3
    ```bash
    sudo apt install libeigen3-dev
    ```
7. assimp
    ```bash
    sudo apt-get install libassimp-dev
    ```
8. cpprestsdk
    ```bash
    sudo apt-get install libcpprest-dev
    ```



## Compilation

We use CMake to configure the program with third-party dependencies and generate the native build system by creating a CMakeLists.txt file. 

1. for collision detection and volume computation:
    ```bash
    cd $src
    mkdir build
    cd build
    cmake ..
    make
    ```

## Model
1. AS: meshes that anatomical structure as the basic unit, e.g., renal pyramid is an anatomical structure and there is a renal_pyramid.off which is the mesh of renal pyramid in off format. 
2. AS_filling_hole: AS after hole filling
3. plain_with_holes: fine-grained meshes that leaf node in glb file as the basic unit, e.g., VH_F_renal_papilla_L_g.off is a basic unit. 
4. plain_filling_hole: plain_with_hole after hole filling
5. ASCT-B_3D_Models_Mapping.csv: famous ASCT-B table
6. organ_origins_meter.csv: origin coordinates of organs generated by extract_origins.py in **scripts** folder. 

## Usage
### Server side: 
1. start http service and receive json request:
    ```bash
    cd $corridor_http_service/build
    ./server2 path_of_3d_model_origins.csv path_of_asct_b.csv body_path path_of_reference_organ_data server_ip port 
    e.g., ./server2 /model/organ_origins_meter_v1.4.csv /model/asct-b-3d-models-crosswalk.csv /model/plain_manifold_filling_hole_v1.4/ /model/reference-organ-data.json 10.0.2.15 12345

    ``` 

    note: 3d_model_origins.csv, asct_b.csv, examples of body_path , and reference-organ-data.json are provided in **model** folder.


### Client side:

POST http://server_ip:port/get-corridor

- JSON request example
```json
{
  "@id": "http://purl.org/ccf/1.5/f31c1726-c693-4734-8c1c-4f1d64bbe034",
  "@type": "SpatialEntity",
  "ccf_annotations": [
      "http://purl.obolibrary.org/obo/UBERON_0002015",
      "http://purl.obolibrary.org/obo/UBERON_0001225",
      "http://purl.obolibrary.org/obo/UBERON_0002189",
      "http://purl.obolibrary.org/obo/UBERON_0000362",
      "http://purl.obolibrary.org/obo/UBERON_0001228",
      "http://purl.obolibrary.org/obo/UBERON_0004200",
      "http://purl.obolibrary.org/obo/UBERON_0001284",
      "http://purl.obolibrary.org/obo/UBERON_0006517",
      "http://purl.obolibrary.org/obo/UBERON_0001227",
      "http://purl.obolibrary.org/obo/UBERON_0008716"
  ],
  "creation_date": "2021-10-03",
  "creator": "Amanda Knoten",
  "creator_first_name": "Amanda",
  "creator_last_name": "Knoten",
  "dimension_units": "millimeter",
  "placement": {
      "@id": "http://purl.org/ccf/1.5/f31c1726-c693-4734-8c1c-4f1d64bbe034_placement",
      "@type": "SpatialPlacement",
      "placement_date": "2021-10-03",
      "rotation_order": "XYZ",
      "rotation_units": "degree",
      "scaling_units": "ratio",
      "target": "http://purl.org/ccf/latest/ccf.owl#VHMLeftKidney",
      "translation_units": "millimeter",
      "x_rotation": 105,
      "x_scaling": 1,
      "x_translation": 50.009,
      "y_rotation": 131,
      "y_scaling": 1,
      "y_translation": 65.904,
      "z_rotation": 44,
      "z_scaling": 1,
      "z_translation": 60.555
  },
  "x_dimension": 5,
  "y_dimension": 6,
  "z_dimension": 2
}
  ```
  - Request as a CURL command on staging server:
  ```bash
  curl -d '@examples/test-corridor.json' -H "Content-Type: application/json" -X POST https://dwwcpwad72.us-east-2.awsapprunner.com/get-corridor -o output_corridor.glb
  ```

  Input: [RUI Registration](examples/test-corridor.json)
  Produces: [result GLB file](examples/sample-corridor-result.glb)
