import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import os

PLOT_AVAILABLE_COLORS = ["blue","orange","green","red","gray","purple","brown","pink","cyan","olive"]
FIGURE_SAVE_PATH = "data/figs"

def extract_saved_result(result:str):
    result = result.replace("(","")
    result = result.replace(")","")
    result = result.replace(" ","")
    mean,std = result.split(",")
    return float(mean),float(std) 

def prepare_figure_dir():
    if not os.path.exists(FIGURE_SAVE_PATH):
        os.mkdir(FIGURE_SAVE_PATH)



def draw_adversarial_average_diagram(adversarials_list,datasets_list,models_list,setting = "evasion",
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True,config="best_config"):
    result_dir_path = f"data/results/{config}"
    plt.figure(figsize=(8, 6))
    for idx,adversarial in enumerate(adversarials_list):
        adversarial_results = {key : [] for key in budgets_list}

        for model in models_list:
            for dataset in datasets_list:
                results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
                results_df = pd.read_csv(results_file_path)
                for budget in budgets_list:
                    row_results = results_df[results_df['budget'] == budget]
                    success_rate = row_results['total'].values[0]
                    mean, _ = extract_saved_result(success_rate)
                    adversarial_results[budget].append(mean)
        
        diagram_y = [stat.mean(adversarial_results[budget]) for budget in budgets_list]
        plt.plot(budgets_list,diagram_y,label=adversarial,color=PLOT_AVAILABLE_COLORS[idx], marker = "o")

    dataset_str = datasets_list[0] if len(datasets_list) == 1 else datasets_list.join(",")
    plt.title(f'Average attack success rate on defense models in {setting} setting on {dataset_str}')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/adverial_perf_average_{setting}_{config}.png")

    if show:
        plt.show()

def draw_adversarial_defense_diagram(adversarials_list,datasets_list,models_list,setting = "evasion",
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True,config = "best_config"):
    result_dir_path = f"data/results/{config}"
    plt.figure(figsize=(8, 6))
    for idx,model in enumerate(models_list):
        adversarial_results = {key : [] for key in budgets_list}
        for adversarial in adversarials_list:
            for dataset in datasets_list:
                results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
                results_df = pd.read_csv(results_file_path)
                for budget in budgets_list:
                    row_results = results_df[results_df['budget'] == budget]
                    success_rate = row_results['total'].values[0]
                    mean, _ = extract_saved_result(success_rate)
                    adversarial_results[budget].append(mean)
        
        diagram_y = [stat.mean(adversarial_results[budget]) for budget in budgets_list]
        plt.plot(budgets_list,diagram_y,label=model,color=PLOT_AVAILABLE_COLORS[idx], marker = "o")

    dataset_str = datasets_list[0] if len(datasets_list) == 1 else datasets_list.join(",")
    plt.title(f'Average attack success rate on defense models in {setting} setting on {dataset_str}')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/defense_perf_average_{setting}_{config}.png")

    if show:
        plt.show()

def draw_adversarial_aggre_diagram(adversarials_list,datasets_list,model,setting = "evasion",aggr_list = ["max","median","sum"],
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True):
    plt.figure(figsize=(8, 6))
    for idx,aggr in enumerate(aggr_list):
        result_dir_path = f"data/results/{aggr}_best_config"

        adversarial_results = {key : [] for key in budgets_list}
        for adversarial in adversarials_list:
            for dataset in datasets_list:
                results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
                results_df = pd.read_csv(results_file_path)
                for budget in budgets_list:
                    row_results = results_df[results_df['budget'] == budget]
                    success_rate = row_results['total'].values[0]
                    mean, _ = extract_saved_result(success_rate)
                    adversarial_results[budget].append(mean)
        
        diagram_y = [stat.mean(adversarial_results[budget]) for budget in budgets_list]
        plt.plot(budgets_list,diagram_y,label=aggr,color=PLOT_AVAILABLE_COLORS[idx], marker = "o")

    dataset_str = datasets_list[0] if len(datasets_list) == 1 else datasets_list.join(",")
    plt.title(f'Average attack success rate on different aggregation of {model} in {setting} setting on {dataset_str}')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/defense_perf_average_{setting}_aggr.png")

    if show:
        plt.show()




if __name__ == "__main__":
    adversarials_list = ["nettack","fga","sga","rnd","pgdattack","prbcdattack","gfattack"]
    # adversarials_list = ["nettack","fga","sga","rnd","prbcdattack","gfattack"]
    datasets_list = ["cora","citeseer","pubmed"]
    models_list = ["GCN","GIN","GSAGE","PNA"]
    aggrs_list = ["sum","mean","max","median"]

    #cora
    datasets_list = ['cora']
    defense_list = ["RobustGCN","GNNGuard","GRAND","ElasticGNN","GCN","GCN_Jaccard","GCN_GARNET","GCN_GTrans"]
    adversarials_list = ["nettack","fga","prbcdattack"]

    # prepare_figure_dir()

    # draw_adversarial_defense_diagram(adversarials_list,datasets_list, defense_list,setting="poision",save_fig=True)

    draw_adversarial_aggre_diagram(adversarials_list,datasets_list,"GSAGE","poision",save_fig=True,aggr_list=aggrs_list)
    
    