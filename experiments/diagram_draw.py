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
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True):
    result_dir_path = "data/results"
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

    plt.title(f'Average attack success rate in {setting} setting')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/adverial_perf_average_{setting}.pdf")

    if show:
        plt.show()




if __name__ == "__main__":
    adversarials_list = ["nettack","fga","sga","rnd","pgdattack","prbcdattack","gfattack"]
    datasets_list = ["cora","citeseer"]
    models_list = ["GCN","GIN","GSAGE","PNA"]
    prepare_figure_dir()

    draw_adversarial_average_diagram(adversarials_list,datasets_list,models_list,setting="evasion")
    
    