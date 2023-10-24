import collections
import urllib.request
import json
import requests
import csv
import os
import shutil
import collections
import pandas as pd

url_collision_detection = "http://localhost:8080/get-collisions"
json_rui_locations = "../data/rui_locations.jsonld"

def send_requests(json_path):

    dic_rui_locations = {}
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        graph = json_data['@graph']
        
        for donor in graph:
            if 'samples' not in donor:
                continue
                
            tissue_blocks = donor['samples']
            for tissue_block in tissue_blocks:
                if 'rui_location' not in tissue_block or not tissue_block['rui_location']:
                    continue
                
                rui_location = tissue_block['rui_location']
                dic_rui_locations[rui_location['@id']] = rui_location
    
    
    for rui_id, rui_location in dic_rui_locations.items():
        post_response = requests.post(url=url_collision_detection, json=rui_location)
    

if __name__ == "__main__":
    
    send_requests(json_rui_locations)




