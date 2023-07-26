'''
Goal:
This program makes a class for species names. I will be using requests from
https://species-names.ivo-bathke.name/suggest/{name} to figure this out.
Given id, link it to a scientific name and a common name.

Categories should be:
- id
- scientific name
- common name

Autocompletion for Dropdowns:
Name autocompletion in dcc.Dropdown is already implemented
given that a list of valid names is provided.

Might also want to do a synonym search if id->scientific name
doesn't give a request with code 200


'''

import requests
import os
from Bio import Entrez
import pickle
import json

def get_dataset_ids(path, descriptor='.txt', specifier="_GC"):
    ids = [id[:-len(descriptor)] for id in os.listdir(path)]
    return ids

def get_scientific_names(ids, specifier="_GC"):
    return [id[:id.index(specifier)].replace('_', ' ') for
            id in ids]
    

def get_common_names(scientific_names, lang='en'):
    '''
    Uses species-names thing
    '''
    req_url = "https://species-names.ivo-bathke.name/name/"
    common_names = dict()
    fails = dict()
    for scientific_name in scientific_names:
        r = requests.get(req_url + scientific_name.replace(' ', '%20')) # %20 is space in HTML URL
        print(scientific_name, r.status_code)
        if r.status_code == 200:
            if r.json().get("common_names"):
                for name in r.json().get("common_names"):
                    if name["lang"] == lang:
                        common_names[scientific_name] = name['name']
        else:
            fails[scientific_name] = ''
    return common_names, fails



def fetch_common_names(scientific_names):
    """
    Usees Biopython Entrez
    
    Fetches common names of species given their scientific names using the NCBI Entrez module.

    Args:
        scientific_names (list): List of scientific names of species.

    Returns:
        dict: A dictionary containing scientific names as keys and their corresponding common names as values.
              If a scientific name is not found or there is an error fetching data, the value will be set to None.
    """
    common_names_dict = {}

    # Set the Entrez email for API usage tracking (replace with your email address).
    Entrez.email = "your_email@example.com"
    for scientific_name in scientific_names:
        try:
            # Use the Entrez ESearch to find the taxonomy ID for the given scientific name.
            search_term = f"{scientific_name}[Scientific Name]"
            esearch_handle = Entrez.esearch(db="taxonomy", term=search_term, retmode="xml")
            search_results = Entrez.read(esearch_handle)
            esearch_handle.close()

            if "IdList" in search_results and len(search_results["IdList"]) > 0:
                taxonomy_id = search_results["IdList"][0]

                # Use the taxonomy ID to fetch the taxonomy record containing the common name.
                esummary_handle = Entrez.esummary(db="taxonomy", id=taxonomy_id, retmode="xml")
                summary_results = Entrez.read(esummary_handle)
                esummary_handle.close()

                common_name = summary_results[0]["CommonName"]
                common_names_dict[scientific_name] = common_name
            else:
                common_names_dict[scientific_name] = None

        except Exception as e:
            # If there's an error during the process, set the value to None and print the error.
            common_names_dict[scientific_name] = None
            print(f"Error fetching data for '{scientific_name}': {e}")

    return common_names_dict


        
if __name__ == '__main__':
    '''Ready to run!'''
    dir = 'msmc_curve_data_birds/"
    path = f'data/{dir}'
    ids = get_dataset_ids(path=path)
    scientific_names = get_scientific_names(ids)
    # print('FETCHING NOW') # Entrez first bc its kinda fast
    fetched_common_names = fetch_common_names(scientific_names)
    # print(fetched_common_names)
    # print("GETTING NOW")
    complement_scientific_names = [k for k in fetched_common_names if fetched_common_names[k]=='']
    # print("complement names list", complement_scientific_names)
    gotten_common_names, fails = get_common_names(complement_scientific_names)
    # print(gotten_common_names)
    # print(fails)
    fetched_common_names.update(gotten_common_names)
    # print("COMBINED")
    # print(fetched_common_names)

    with open(f"{dir[:-1]}_common_names.json", "w") as outfile:
        json.dump(fetched_common_names, outfile)
        
    with open(f"{dir[:-1]}_failed_common_names.json", "w") as outfile:
        json.dump(fails, outfile)