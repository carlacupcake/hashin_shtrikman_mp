## Libraries and packages needed to run with MP-API
'''
boto3
custodian
emmet-core[all]>=0.39.8
monty==2022.3.12
mpcontribs-client>=5.0.10
mp_api
msgpack
pydantic>=1.8.2
pymatgen>=2022.3.7
pymatgen-analysis-alloys>=0.0.3
typing-extensions==4.1.1
requests==2.27.1
'''
import json
from mp_api.client import MPRester
from mpcontribs.client import Client
from emmet.core.summary import HasProps

## User Input
get_band_gap = False
get_bulk_modulus = True
get_e_electronic = True
get_e_ij_max = True
get_e_ionic = True
get_e_total = True
get_elec_cond = False
get_n = True
get_shear_modulus  = True  
get_therm_cond = False
get_total_magnetization = False
get_total_magnetization_normalized_vol = False
get_total_magnetization_normalized_formula_units = False
get_universal_anisotropy = False

get_mp_ids_contrib = (get_elec_cond and get_therm_cond)

from datetime import datetime




def generate_json(get_band_gap,
                  get_bulk_modulus, 
                  get_e_electronic, 
                  get_e_ij_max, 
                  get_e_ionic, 
                  get_e_total, 
                  get_elec_cond, 
                  get_mp_ids_contrib, 
                  get_n, 
                  get_shear_modulus, 
                  get_therm_cond, 
                  get_total_magnetization, 
                  get_total_magnetization_normalized_formula_units, 
                  get_total_magnetization_normalized_vol, 
                  get_universal_anisotropy):

    if get_mp_ids_contrib:

        client = Client(apikey="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM", project="carrier_transport")
        '''
        list_of_query_params = client.available_query_params()
        for item in list_of_query_params:
            print(item)  # print list of available query parameters
        '''
    else:
        client = Client(apikey="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM")

    ## Assemble dictionary of values needed for Hashin-Shtrikman analysis
    final_dict = {"mp-ids": [],
                "mp-ids-contrib": [], 
                "formula": [],
                "metal": [],
                "bulk_modulus": [],
                "shear_modulus": [],
                "universal_anisotropy": [],
                "e_total": [],
                "e_ionic": [],
                "e_electronic": [],
                "n": [],
                "e_ij_max": [],
                "therm_cond_300K_low_doping": [],
                "elec_cond_300K_low_doping": []}
    
    fields = ["material_id", "is_stable", "is_metal"]
    if get_band_gap:
        fields.append("band_gap")
    if get_bulk_modulus:
        fields.append("bulk_modulus")
    if get_e_electronic:
        fields.append("e_electronic")
    if get_e_ij_max:
        fields.append("e_ij_max")
    if get_e_ionic:
        fields.append("e_ionic")
    if get_e_total:
        fields.append("e_total")
    if get_n:
        fields.append("n")
    if get_shear_modulus:
        fields.append("shear_modulus")
    if get_total_magnetization:
        fields.append("total_magnetization")
    if get_total_magnetization_normalized_formula_units:
        fields.append("total_magnetization_normalized_formula_units")
    if get_total_magnetization_normalized_vol:
        fields.append("total_magnetization_normalized_vol")
    if get_universal_anisotropy:
        fields.append("universal_anisotropy")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with MPRester("uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM") as mpr:
        docs = mpr.materials.summary.search(fields=fields)
        '''
        list_of_available_fields = mpr.materials.summary.available_fields
        for item in list_of_available_fields:
            print(item)  # print list of available field parameters
        '''
    
        #mp_ids = []

        # Calculate the size of each chunk
        chunk_size = len(docs) // size
        # chunk_size = 100 // size

        # Calculate the start and end indices for this process's chunk
        start = rank * chunk_size
        end = start + chunk_size if rank != size - 1 else len(docs)  # The last process gets the remainder
        # end = start + chunk_size if rank != size - 1 else 100  # The last process gets the remainder

        # Each process gets a different chunk
        chunk = docs[start:end]

        # for i, doc in enumerate(docs):
        # for i, doc in enumerate(docs[0:100]):
        for i, doc in enumerate(chunk):
        # for i, doc in enumerate(chunk[0:100]):

            # print(f"{i} of {len(docs[0:100])}")
            print(f"Process {rank}: {i} of {len(chunk)}")
            # print(f"Process {rank}: {i} of {len(chunk[0:100])}")

            try:
                # required_fields = [doc.material_id, doc.is_stable, doc.is_metal, 
                #                    doc.bulk_modulus, doc.shear_modulus, 
                #                    doc.universal_anisotropy, doc.e_electronic, 
                #                    doc.e_ij_max, doc.e_ionic, 
                #                    doc.e_total, doc.n, doc.total_magnetization, 
                #                    doc.total_magnetization_normalized_formula_units, 
                #                    doc.total_magnetization_normalized_vol]
                # required_fields = [doc.material_id, doc.is_stable, doc.is_metal, 
                #                    doc.bulk_modulus, doc.shear_modulus, 
                #                    doc.universal_anisotropy, doc.e_electronic, 
                #                    doc.e_ij_max, doc.e_ionic, 
                #                    doc.e_total, doc.n]
                required_fields = [doc.material_id, doc.is_stable, doc.is_metal, 
                                   doc.bulk_modulus, doc.shear_modulus]
                
                if all(field is not None for field in required_fields):

                    mp_id = doc.material_id                           
                    query = {"identifier": mp_id}
                    my_dict = client.download_contributions(query=query, include=["tables"])[0]
                    
                    required_fields_contrib = [my_dict["identifier"]]

                    if all(field is not None for field in required_fields_contrib):
                        final_dict["mp-ids"].append(mp_id)    
                        final_dict["formula"].append(my_dict["formula"])
                        final_dict["metal"].append(my_dict["data"]["metal"])                  
                        final_dict["is_stable"].append(doc.is_stable)
                        final_dict["is_metal"].append(doc.is_metal) 

                        if get_band_gap:
                            final_dict["band_gap"].append(doc.band_gap)
                        if get_bulk_modulus:
                            bulk_modulus_voigt = doc.bulk_modulus["voigt"]
                            final_dict["bulk_modulus"].append(bulk_modulus_voigt)
                        if get_e_electronic:
                            final_dict["e_electronic"].append(doc.e_electronic)
                        if get_e_ij_max:
                            final_dict["e_ij_max"].append(doc.e_ij_max)
                        if get_e_ionic:
                            final_dict["e_ionic"].append(doc.e_ionic)
                        if get_e_total:
                            final_dict["e_total"].append(doc.e_total)
                        if get_n:
                            final_dict["n"].append(doc.n)
                        if get_shear_modulus:
                            shear_modulus_voigt = doc.shear_modulus["voigt"]
                            final_dict["shear_modulus"].append(shear_modulus_voigt) 
                        if get_total_magnetization:
                            final_dict["total_magnetization"].append(doc.total_magnetization)
                        if get_total_magnetization_normalized_formula_units:
                            final_dict["total_magnetization_normalized_formula_units"].append(doc.total_magnetization_normalized_formula_units)
                        if get_total_magnetization_normalized_vol:
                            final_dict["total_magnetization_normalized_vol"].append(doc.total_magnetization_normalized_vol)
                        if get_universal_anisotropy:
                            final_dict["universal_anisotropy"].append(doc.universal_anisotropy)

                        if get_mp_ids_contrib:

                            try:
                                final_dict["mp-ids-contrib"].append(my_dict["identifier"])
                                thermal_cond = my_dict["tables"][7].iloc[2, 1] * 1e-14  # multply by relaxation time, 10 fs
                                elec_cond = my_dict["tables"][5].iloc[2, 1] * 1e-14 # multply by relaxation time, 10 fs   
                                final_dict["therm_cond_300K_low_doping"].append(thermal_cond)
                                final_dict["elec_cond_300K_low_doping"].append(elec_cond)              

                            except:
                                IndexError

            except:
                TypeError
                print(f"TypeError. One or more requested fields are not available for this material with mp-id = {doc.material_id}.")
                # list_of_available_fields = mpr.materials.summary.available_fields
                # for item in list_of_available_fields:
                #     print(item)  # print list of available field parameters
                # print(f"doc = {doc}")
                # print(f"doc_bm = {doc.bulk_modulus}")
                # print(doc.bulk_modulus.get('voigt'))
                # print(doc.bulk_modulus["voigt"])
                # print(doc.e_ij_max)

    # After the for loop
    final_dicts = comm.gather(final_dict, root=0)

    # On process 0, consolidate the results
    if rank == 0:
        consolidated_dict = {"mp-ids": [],
                "mp-ids-contrib": [], 
                "formula": [],
                "metal": [],
                "bulk_modulus": [],
                "shear_modulus": [],
                "universal_anisotropy": [],
                "e_total": [],
                "e_ionic": [],
                "e_electronic": [],
                "n": [],
                "e_ij_max": [],
                "therm_cond_300K_low_doping": [],
                "elec_cond_300K_low_doping": []}

        for final_dict in final_dicts:
            for key in consolidated_dict:
                consolidated_dict[key].extend(final_dict[key])

        # Save the consolidated results to a JSON file
        now = datetime.now()
        my_file_name = "final_dict_test_" + now.strftime("%m_%d_%Y_%H_%M_%S")
        with open(my_file_name, "w") as my_file:
            json.dump(consolidated_dict, my_file)


    # with open("final_dict_test.json", "w") as my_file:
    #     json.dump(final_dict, my_file)

    return


generate_json(get_band_gap,
            get_bulk_modulus, 
            get_e_electronic, 
            get_e_ij_max, 
            get_e_ionic, 
            get_e_total, 
            get_elec_cond, 
            get_mp_ids_contrib, 
            get_n, 
            get_shear_modulus, 
            get_therm_cond, 
            get_total_magnetization, 
            get_total_magnetization_normalized_formula_units, 
            get_total_magnetization_normalized_vol, 
            get_universal_anisotropy)