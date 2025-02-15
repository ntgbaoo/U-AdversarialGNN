import pandas as pd
import numpy
import statistics as stat

RESULT_PATH = "data/results"


def generate_summary_table(config,seting,dataset,model,adversarial_lists,save= True, no_degree = False):
    if no_degree:
        result_dir_path = f"data/results/ablation/no_degree/{config}"
    else:
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
    new_dict = {}
    for key in dict.keys():
        curr_arr = dict[key]
        
        new_dict[key] = numpy.array([float(element.split(",")[0]) for element in curr_arr])
    return new_dict
    


def calc_difference_default_vs_best(seting,dataset,model,adversarial_lists,save=True):
    ori_default = generate_summary_table("default",seting,dataset,model,adversarial_lists,save=False)
    ori_best_config = generate_summary_table("best_config",seting,dataset,model,adversarial_lists,save=False)
    ori_best_config.pop("budget")
    ori_default.pop("budget")
    default = retrieve_only_mean(ori_default)
    best_config = retrieve_only_mean(ori_best_config)

    result_dict = {}
    result_dict['budget'] = [1,2,3,4,5]
    for adversarial in default.keys():
        difference = numpy.round((best_config[adversarial] - default[adversarial]),decimals=3)
        col = []
        for idx,value in enumerate(difference):
            col.append(f"{ori_default[adversarial][idx]},{value}")
        result_dict[adversarial] = col

    if save:
        df = pd.DataFrame(result_dict)
        df.to_csv(f"data/results/weekly_meeting_tabels/config_difference_{dataset}_{model}_{seting}.csv") 
    return result_dict

def calc_difference_no_degree_vs_best(seting,dataset,model,adversarial_lists,save=True):
    ori_default = generate_summary_table("default",seting,dataset,model,adversarial_lists,save=False,no_degree=True)
    ori_best_config = generate_summary_table("best_config",seting,dataset,model,adversarial_lists,save=False)
    ori_best_config.pop("budget")
    ori_default.pop("budget")
    default = retrieve_only_mean(ori_default)
    best_config = retrieve_only_mean(ori_best_config)

    result_dict = {}
    result_dict['budget'] = [1,2,3,4,5]
    for adversarial in default.keys():
        difference = numpy.round((best_config[adversarial]-default[adversarial]),decimals=3)
        col = []
        for idx,value in enumerate(difference):
            col.append(f"{ori_default[adversarial][idx]},{value}")
        result_dict[adversarial] = col

    if save:
        df = pd.DataFrame(result_dict)
        df.to_csv(f"data/results/weekly_meeting_tabels/config_difference_{dataset}_{model}_{seting}_no_degree.csv") 
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
    
def extract_saved_result(result:str):
    result = result.replace("(","")
    result = result.replace(")","")
    result = result.replace(" ","")
    mean,std = result.split(",")
    return float(mean),float(std) 
    
def compute_diff_between_attack(attack, compared_attack,config_setting="best_config",settings_list = ["evasion","poision"], models_list= ["GCN"],datasets_list = ['cora']):
    differences = []
    for setting in settings_list:
        for budget in [1,2,3,4,5]:
            for model in models_list:
                for dataset in datasets_list:
                    result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{attack}_{setting}.csv"
                    resulf_df = pd.read_csv(result_file_path)
                    results = resulf_df['total'].tolist()
                    assert len(results) == 5
                    mean,std = extract_saved_result(results[budget-1])

                    result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{compared_attack}_{setting}.csv"
                    resulf_df = pd.read_csv(result_file_path)
                    results = resulf_df['total'].tolist()
                    assert len(results) == 5
                    compared_mean,compared_std = extract_saved_result(results[budget-1])
                    # print(mean,compared_mean)
                    differences.append(mean - compared_mean)
    
    return stat.mean(differences) 

                        
if __name__ == "__main__":
    config = "default"
    setting = "poision"
    dataset = "cora"
    model= 'PNA'
    adversarial_lists = ['rnd','fga','nettack','pgdattack','prbcdattack','sga','gfattack']

    # result = calc_difference_evasion_vs_poision(config,dataset,model,adversarial_lists,save=True)
    # print(result)

    result_dict = calc_difference_no_degree_vs_best("poision","cora","PNA",adversarial_lists)
    print(result_dict)
    # generate_summary_table("default","poision","citeseer","PNA",adversarial_lists,True)


    # adversarials_list = ['random_degree_attack','rnd']
    # models_list = ['GCN',"GIN","GSAGE","PNA"]
    # defense_models = ['GCN_GARNET',"GNNGuard","GRAND"]
    # datasets_list = ['cora','citeseer','pubmed']

    # print(compute_diff_between_attack(adversarials_list[0],adversarials_list[1],settings_list = ['evasion','poision'],models_list=['GRAND'],datasets_list=datasets_list))
    
    