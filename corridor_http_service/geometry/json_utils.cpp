#include "json_utils.h"


void load_organ_transformation(const std::string &json_file_path, std::unordered_map<std::string, Placement> &mapping_placement)
{
    json::value json_output = read_json_file(json_file_path);

    try 
    {
        auto json_output_object = json_output.as_object();
        auto placement = json_output_object.at("placementPatches");
        auto placement_object = placement.as_object();
        
        for (auto it=placement_object.cbegin(); it!=placement_object.cend(); ++it)
        {
            // auto &source = (it->first).as_string();
            auto &propertyValue = it->second;

            std::string source = propertyValue.at("source").as_string();
            std::string target = propertyValue.at("target").as_string();
            double x_scaling = propertyValue.at("x_scaling").as_double();
            double y_scaling = propertyValue.at("y_scaling").as_double();
            double z_scaling = propertyValue.at("z_scaling").as_double();
            double x_translation = propertyValue.at("x_translation").as_double();
            double y_translation = propertyValue.at("y_translation").as_double();
            double z_translation = propertyValue.at("z_translation").as_double();
            double x_rotation = propertyValue.at("x_rotation").as_double();
            double y_rotation = propertyValue.at("y_rotation").as_double();
            double z_rotation = propertyValue.at("z_rotation").as_double();
            

            mapping_placement[source] = Placement(source, target, 
                                                            x_scaling, y_scaling, z_scaling,
                                                            x_translation, y_translation, z_translation,
                                                            x_rotation, y_rotation, z_rotation);
        }

        // for (int i = 0; i < placement.size(); i++) 
        // {
        //     auto item = placement[i];
        //     auto itemObj = item.as_object();
        //     for (auto iterInner = itemObj.cbegin(); iterInner != itemObj.cend(); ++iterInner)
        //     {
        //         auto &propertyValue = iterInner->second;
        //         std::string source = propertyValue.at("source").as_string();
        //         std::string target = propertyValue.at("target").as_string();
        //         double x_scaling = propertyValue.at("x_scaling").as_double();
        //         double y_scaling = propertyValue.at("y_scaling").as_double();
        //         double z_scaling = propertyValue.at("z_scaling").as_double();
        //         double x_translation = propertyValue.at("x_translation").as_double();
        //         double y_translation = propertyValue.at("y_translation").as_double();
        //         double z_translation = propertyValue.at("z_transaltion").as_double();
        //         double x_rotation = propertyValue.at("x_rotation").as_double();
        //         double y_rotation = propertyValue.at("y_rotation").as_double();
        //         double z_rotation = propertyValue.at("z_rotation").as_double();

        //         mapping_placement[source] = Placement(source, target, x_scaling, y_scaling, z_scaling, x_translation, y_translation, z_translation, x_rotation, y_rotation, z_rotation);

        //     }
        // }



    }
    catch(...) 
    {
        std::cout << "catch exception in parse " + json_file_path << std::endl;
    }


}

json::value read_json_file(std::string const & json_file_path)
{
    json::value output;  // JSON read from input file

    try
    {
        // Open the file stream
        std::ifstream f(json_file_path);
        // String stream for holding the JSON file
        std::stringstream strStream;

        // Stream file stream into string stream
        strStream << f.rdbuf();

        // std::cout << strStream.str() << std::endl;
        f.close();  // Close the filestream

        // Parse the string stream into a JSON object
        output = json::value::parse(strStream);
    }
    catch (json::json_exception excep)
    {
        throw json::json_exception("Error Parsing JSON file " + json_file_path);
    }

    return output;
}