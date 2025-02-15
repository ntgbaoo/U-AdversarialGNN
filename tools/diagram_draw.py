import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import os
import numpy as np

PLOT_AVAILABLE_COLORS = ["blue","orange","green","red","gray","purple","brown","pink","cyan","olive"]
FIGURE_SAVE_PATH = "data/figs"
MARKER = ["o","v","s","P","*","X","D","^",">","<"]

adversarial_name = {
    'rnd' : 'Random',
    'fga': 'FGA',
    'nettack':'NETTACK',
    'pgdattack': ' PGD',
    'prbcdattack' : 'PR-BCD',
    'sga' :'SGA',
    'gottack' : 'GOttack',
    'random_degree_attack': 'I1D-RND' 
}


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
        plt.plot(budgets_list,diagram_y,label=adversarial,color=PLOT_AVAILABLE_COLORS[idx], marker = MARKER[idx])

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
        plt.plot(budgets_list,diagram_y,label=model,color=PLOT_AVAILABLE_COLORS[idx],marker = MARKER[idx],markersize=8)

    dataset_str = datasets_list[0] if len(datasets_list) == 1 else datasets_list.join(",")
    # plt.title(f'Average attack success rate on defense models in {setting} setting on {dataset_str}')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/defense_perf_average_{setting}_{config}_{dataset_str}.pdf")

    if show:
        plt.show()

def draw_adversarial_aggre_diagram(adversarials_list,datasets_list,model,setting = "evasion",aggr_list = ["max","median","sum"],
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True,column = "total"):
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
                    success_rate = row_results[column].values[0]
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
        plt.savefig(f"{FIGURE_SAVE_PATH}/defense_perf_average_{setting}_aggr_{column}.png")

    if show:
        plt.show()


def draw_node_category(adversarials_list,datasets_list,setting = "evasion",config = "best_config",models_list = ["GCN","GIN","GSAGE","PNA"] , 
                                     budgets_list = [1,2,3,4,5],save_fig = True, show = True,column = "total"):


    plt.figure(figsize=(4, 4))
    for idx,model in enumerate(models_list):
        result_dir_path = f"data/results/{config}"

        adversarial_results = {key : [] for key in budgets_list}
        for adversarial in adversarials_list:
            for dataset in datasets_list:
                results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
                results_df = pd.read_csv(results_file_path)
                for budget in budgets_list:
                    row_results = results_df[results_df['budget'] == budget]
                    success_rate = row_results[column].values[0]
                    mean, _ = extract_saved_result(success_rate)
                    adversarial_results[budget].append(mean)
        
        diagram_y = [stat.mean(adversarial_results[budget]) for budget in budgets_list]
        plt.plot(budgets_list,diagram_y,label=model,color=PLOT_AVAILABLE_COLORS[idx], marker = MARKER[idx])

    dataset_str = datasets_list[0] if len(datasets_list) == 1 else ",".join(datasets_list)
    node_cat = column.replace("_"," ")
    # plt.title(f'Average miss-classification rate of {node_cat} nodes in {setting} setting on {dataset_str}')
    plt.xlabel('Budget')
    plt.ylim(0, 1)
    plt.xticks(budgets_list)
    plt.ylabel('Attack success rate (%)')
    plt.legend()

    if save_fig:
        plt.savefig(f"{FIGURE_SAVE_PATH}/node_cat_{column}_{setting}_{config}.pdf")

    if show:
        plt.show()


def draw_meta_fig(adversarials_list,dataset,setting = "evasion",config = "best_config",model = "GCN", budget = 1,save_fig = True, show = True,column = "total",legend = True, yaxis = True):
    config_means = []
    config_stds = []
    default_means = []
    default_stds = []
    no_degree_means = []
    no_degree_stds = []

    for adversarial in adversarials_list:
        # result for config
        result_dir_path = f"data/results/{config}"
        results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
        results_df = pd.read_csv(results_file_path)
        row_results = results_df[results_df['budget'] == budget]
        success_rate = row_results[column].values[0]
        mean, std = extract_saved_result(success_rate)
        config_means.append(mean)
        config_stds.append(std)

        # result for compared_config
        result_dir_path = f"data/results/default"
        results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
        results_df = pd.read_csv(results_file_path)
        row_results = results_df[results_df['budget'] == budget]
        success_rate = row_results[column].values[0]
        mean, _ = extract_saved_result(success_rate)
        default_means.append(mean)
        default_stds.append(std)

        result_dir_path = f"data/results/ablation/no_degree/best_config"
        results_file_path = f"{result_dir_path}/{model}/{dataset}/{adversarial}_{setting}.csv"
        results_df = pd.read_csv(results_file_path)
        row_results = results_df[results_df['budget'] == budget]
        success_rate = row_results[column].values[0]
        mean, _ = extract_saved_result(success_rate)
        no_degree_means.append(mean)
        no_degree_stds.append(std)
    
    # Create the figure and axes
    diagram_width =10
    width = diagram_width + 0.2
    if yaxis:
        width += 1
    if legend:
        width += 4.8
    fig, ax = plt.subplots(figsize=(width, 8))

    # Define width for each bar
    bar_width = 0.5


    # Calculate x-positions for each bar group
    x_positions = np.arange(len(adversarials_list))*2

    # Plot the first set of bars
    ax.bar(x_positions, default_means, bar_width, label='w/o model selection', color='orange',alpha=0.7,yerr= default_stds)

    # Plot the second set of bars, shifted to the right
    ax.bar(x_positions + bar_width, config_means, bar_width, label='Our evaluation', color='blue',alpha=0.7, yerr = config_stds)

    # Plot the third set of bars, shifted further to the right
    ax.bar(x_positions + 2 * bar_width, no_degree_means, bar_width, label='w/o node degree', color='green',alpha=0.7,yerr = no_degree_stds)

    # Set y-ticks and labels
    ax.set_xticks(x_positions + bar_width)  # Center y-ticks between bars
    for i in range(len(adversarials_list)):
        adversarials_list[i] = adversarial_name[adversarials_list[i]]
    
    if yaxis:
        ax.set_ylabel('Misclassification Rate',fontsize=28)

    ax.set_xticklabels(adversarials_list,fontsize=25)

    # Add labels and title
    ax.set_ylim(0, 1.0) 
    ax.tick_params(axis='y', labelsize=26)

    # ax.set_ylabel('Adversarial attacks')
    # ax.set_title('Adversarial Accuracy vs. Correct Predictions')

    # Add legend
    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',borderaxespad=0.,fontsize = 25) 
    
    # ax.legend() 


    # Adjust spacing between bars
    if yaxis:
        left = 0.10
    else:
        ax.set_yticklabels([]) 
        left = 0.01
    right = (diagram_width/width) + left
    print(right)
    plt.subplots_adjust(left=left, right=right)  # Adjust as needed
    # Show the plot
    if save_fig:
            plt.savefig(f"{FIGURE_SAVE_PATH}/meta_{dataset}_budget{budget}_{model}_{setting}.pdf")

    if show:
        plt.show()
   



if __name__ == "__main__":
    adversarials_list = ["nettack","fga","sga","rnd","pgdattack","prbcdattack"]
    # adversarials_list = ["nettack","fga","sga","rnd","prbcdattack","gfattack"]
    datasets_list = ["cora","citeseer","pubmed"]
    models_list = ["GCN","GIN","GSAGE","PNA"]
    aggrs_list = ["sum","mean","max","median"]

    #cora
    datasets_list = ['cora','citeseer']
    defense_list = ["RobustGCN","GNNGuard","GRAND","ElasticGNN","GCN_Jaccard","GCN_GARNET"]

    # prepare_figure_dir()

    datasets_list = ['pubmed']
    defense_adversarial = {
    'cora' : ['nettack','prbcdattack','gottack'],
    'citeseer':['nettack','prbcdattack','fga'],
    'pubmed':['nettack','fga','gottack']
    }
    # print(len(defense_list) + len(models_list))

    # draw_adversarial_defense_diagram(defense_adversarial['pubmed'],datasets_list, defense_list + ['GSAGE'],setting="evasion",save_fig=True,show=False)


    # draw_adversarial_aggre_diagram(adversarials_list,datasets_list,"GSAGE","evasion",save_fig=True,aggr_list=aggrs_list,column="other",show=False)
    

    # draw_node_category(adversarials_list,datasets_list=datasets_list,setting="poision",save_fig=True,column="other",show=False,config="default")

    adversarials_list = ["random_degree_attack","fga","nettack","pgdattack","prbcdattack","sga","gottack"]
    draw_meta_fig(adversarials_list[::-1],"pubmed",config="best_config",model="GSAGE",budget=5,save_fig=True,show=True, legend = True, yaxis = False,setting="poision")