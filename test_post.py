# created by Lu Chen
# date: June 14, 2022

import json
import requests
import urllib.request
import csv

collision_detection_url = 'http://0.0.0.0:12345/get-corridor'
enriched_rui_location_url = 'https://cdn.humanatlas.io/digital-objects/ds-graph/hra-pop-full/v0.5.1/assets/full-dataset-graph.jsonld'
                            
responses = []

fields = ['rui_location_id', 'parallel_time']
rows = []

csvfile = open('result.txt', 'w')
writer = csv.writer(csvfile)
writer.writerow(fields)

with urllib.request.urlopen(enriched_rui_location_url) as url:
    data = json.load(url)
    graph = data['@graph']

    for person in graph:
        samples = person['samples']
        for sample in samples:
            if sample['sample_type'] == 'Tissue Block':
                if 'rui_location' in sample:
                    rui_location = sample['rui_location']
                    r = requests.post(url=collision_detection_url, json=rui_location)
                    #response = {'@id': rui_location['@id'], 'result': r.json()}
                    rui_location_id = rui_location['@id']
                    if r.json() and 'parallel_time' in r.json(): 
                        time_eclipse = r.json()['parallel_time']
                    #rows.append([rui_location_id, time_eclipse]) 
                        writer.writerow([rui_location_id, time_eclipse])


                #responses.append(response)

# with open('result.txt', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(fields)
#     writer.writerows(rows)

#print(responses)
# with open('collision_result.json', 'w') as write_file:
#     json.dump(responses, write_file, indent=4)