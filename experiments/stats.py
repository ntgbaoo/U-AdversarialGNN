import pandas as pd
import numpy

def generate_summary_table(config,seting,dataset,model,adversarial_lists,save= True):
    result_dir_path = f"data/results/{config}"
    result_dict = {}
    result_dict['budget'] = [1,2,3,4,5]

    for adversarial in adversarial_lists:
        results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{seting}.csv"
        result_df = pd.read_csv(results_file_path)
        budget_results= result_df['total'].tolist()
        for i in range(len(budget_results)):
            curr_value = budget_results[i].replace("(","").replace(")","")
            curr_value = curr_value.split(",")
            curr_value= [str(round(float(element),3)) for element in curr_value]
            curr_value = ",".join(curr_value)
            budget_results[i] = curr_value
            
        result_dict[adversarial] = budget_results
    
    if save:
        df = pd.DataFrame(result_dict)
        df.to_csv(f"data/results/weekly_meeting_tabels/{dataset}_{model}_{seting}_{config}.csv")  
    return result_dict

def retrieve_only_mean(dict:dict):
    for key in dict.keys():
        curr_arr = dict[key]
        
        dict[key] = numpy.array([float(element.split(",")[0]) for element in curr_arr])
    


def calc_difference_default_vs_best(seting,dataset,model,adversarial_lists,save=True):
    default = generate_summary_table("default",seting,dataset,model,adversarial_lists,save=False)
    best_config = generate_summary_table("best_config",seting,dataset,model,adversarial_lists,save=False)
    best_config.pop("budget")
    default.pop("budget")
    retrieve_only_mean(default)
    retrieve_only_mean(best_config)

    result_dict = {}
    result_dict['budget'] = [1,2,3,4,5]
    for adversarial in default.keys():
        result_dict[adversarial] = numpy.round((best_config[adversarial] - default[adversarial]),decimals=3)

    if save:
        df = pd.DataFrame(result_dict)
        df.to_csv(f"data/results/weekly_meeting_tabels/config_difference_{dataset}_{model}_{seting}.csv") 
    return result_dict

def calc_difference_evasion_vs_poision(config,dataset,model,adversarial_lists,save=True):
    evasion = generate_summary_table(config,"evasion",dataset,model,adversarial_lists,save=False)
    poision = generate_summary_table(config,"poision",dataset,model,adversarial_lists,save=False)
    evasion.pop("budget")
    poision.pop("budget")
    retrieve_only_mean(evasion)
    retrieve_only_mean(poision)

    result_dict = {}
    result_dict['budget'] = [1,2,3,4,5]
    for adversarial in poision.keys():
        result_dict[adversarial] = numpy.round((poision[adversarial] - evasion[adversarial]),decimals=3)

    if save:
        df = pd.DataFrame(result_dict)
        df.to_csv(f"data/results/weekly_meeting_tabels/setting_difference_{dataset}_{model}_{config}.csv") 
    return result_dict
    
    
if __name__ == "__main__":
    config = "default"
    setting = "poision"
    dataset = "cora"
    model= 'PNA'
    adversarial_lists = ['rnd','fga','nettack','pgdattack','prbcdattack','sga','gfattack']

    # result = calc_difference_evasion_vs_poision(config,dataset,model,adversarial_lists,save=True)
    # print(result)

    calc_difference_default_vs_best(setting,"cora","GIN",adversarial_lists)
    
    