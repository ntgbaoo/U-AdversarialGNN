import pandas as pd
import statistics as stat
from pprint import pprint
RESULT_PATH = "data/results"

defense_adversarial = {
    'cora' : ['nettack','prbcdattack','gottack'],
    'citeseer':['nettack','prbcdattack','fga'],
    'pubmed':['nettack','fga','gottack']
}

adversarial_name = {
    'rnd' : 'Random',
    'fga': 'FGA',
    'nettack':'NETTACK',
    'pgdattack': ' PGD',
    'prbcdattack' : 'PR-BCD',
    'sga' :'SGA',
    'gottack' : 'GOttack',
    'random_degree_attack': '\\propseRND' 
}

setting_dict = {
    "evasion": "Evasion",
    "poision": "Poison"
}

def extract_saved_result(result:str):
    result = result.replace("(","")
    result = result.replace(")","")
    result = result.replace(" ","")
    mean,std = result.split(",")
    return float(mean),float(std) 

def generate_meta_table(budget,config_setting,no_degree = False,adversarials_list = ['nettack'], models_list= ["GCN"],datasets_list = ['cora']):
    r"""
    Generate table in format as follow:

    \\begin{table*}[tbp]
    \\centering
    \\caption{Misclassification rate (in $\%$) ($\uparrow$) of target nodes in four datasets where four backbone GNNs (GCN, GIN, GraphSAGE and PNA) are attacked in node classification with budget $\Delta=5$ in evasion setting.}
    \\label{tab:missclass-table}
        resizebox{\linewidth}{!}{ 
        \begin{tabular}{cc | cccc | cccc | cccc |c }
        \toprule 
        &\multicolumn{1}{c}{Dataset}& \multicolumn{4}{c}{Cora} & \multicolumn{4}{c}{Citeseer}  &  \multicolumn{4}{c}{Pubmed} &   \\ \midrule
        &Method  & GCN & GIN & GSAGE & PNA & GCN & GIN & GSAGE & PNA & GCN & GIN & GSAGE & PNA  & Ranking\\ \midrule
        \multirow{7}{*}{\rotatebox{90}{\textbf{Evasion}}} &Random & 30.04 & 37.73 & 37.60 & 39.07 & 36.00 & 37.47 & 35.07 & 46.40 & 15.07 & 19.33 & 13.07 & 12.93\\
            &FGA & \underline{64.67} & \underline{55.47} & \underline{55.33} & \underline{62.80} & 58.27 & \underline{55.33} & \underline{57.87} & \underline{56.13} & \underline{58.00} & \underline{55.47} & \underline{42.40} & \underline{50.80}\\ 
            &Nettack & 63.73 & \textbf{65.20} & \textbf{63.73} & \textbf{68.80} & \underline{59.73} & \textbf{62.53} & \textbf{58.53} & \textbf{69.60} & \textbf{58.67} & \textbf{55.87} & \textbf{42.53} & \textbf{52.80}\\
            &PGD & 59.20 & 47.60 & 46.13 & 48.40 & 55.47 & 45.60 & 45.33 & 42.53 & 44.93 & 39.87 & 32.00 & 30.93\\
            &PR-BCD & \textbf{65.73} & 54.13 & 51.07 & 52.13 & \textbf{65.87} & 49.60 & 53.60 & 53.07 & 54.13 & 36.13 & 28.80 & 28.53\\
            &SGA & 60.67 & 51.20 & 49.47 & 50.53 & 53.37 & 47.60 & 45.20 & 48.13 & 54.80 & 50.27 & 36.53 & 44.93\\
        \hline

    """
    columns = 'cc | ' + ' | '.join(['c'* len(models_list)] * len(datasets_list)) + ' | c'

    table_head = f"&\\multicolumn{{1}}{{c}}{{Dataset}}"
    for dataset in datasets_list:
        table_head += f"& \\multicolumn{{{len(models_list)}}}{{c}}{{{dataset}}}"

    table_head += "& \\\\ \\midrule"
    
    output = f"""
    \\begin{{table*}}[tbp]
    \\centering
    \\caption{{Misclassification rate (in $\\%$) ($\\uparrow$) of target nodes in four datasets where four backbone GNNs (GCN, GIN, GraphSAGE and PNA) are attacked in node classification with budget $\\Delta={budget}$.}}
    \\label{{tab:missclass-table}}
        \\resizebox{{\\linewidth}}{{!}}{{ 
        \\begin{{tabular}}{{ {columns} }}
        \\toprule 
        {table_head}
        &Method  & GCN & GIN & GSAGE & PNA & GCN & GIN & GSAGE & PNA & GCN & GIN & GSAGE & PNA  & Ranking\\\\ \\midrule
    """
    for setting in ["evasion","poision"]:
        adversarial_rank = {}
        best_second_record = {}
        for dataset in datasets_list:
            for model in models_list:
                adversarial_performance = []
                for adversarial in adversarials_list:
                    result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                    resulf_df = pd.read_csv(result_file_path)
                    results = resulf_df['total'].tolist()
                    assert len(results) == 5
                    mean,std = extract_saved_result(results[budget-1])

                    adversarial_performance.append((adversarial,mean))
                adversarial_sort = [x[0] for x in sorted(adversarial_performance,reverse=True, key=lambda item: item[1])]
                for idx,sort_adv in enumerate(adversarial_sort):
                    if not sort_adv in adversarial_rank:
                        adversarial_rank[sort_adv]= []
                    adversarial_rank[sort_adv].append(idx + 1)
                
                if not model in best_second_record:
                    best_second_record[model] = {}
                
                best_second_record[model][dataset] = adversarial_sort[:2]
        
        for key in adversarial_rank.keys():
            adversarial_rank[key] = stat.mean(adversarial_rank[key])

        multi_row = len(adversarials_list)
        for idx,adversarial in enumerate(adversarials_list):
            if idx == 0:
                row = f"""
                \\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{setting_dict[setting]}}}}}}} & {adversarial_name[adversarial]}
                """
            else:
                row = f"""
                & {adversarial_name[adversarial]}
                """

            for dataset in datasets_list:
                for model in models_list:
                    best = best_second_record[model][dataset][0]
                    second = best_second_record[model][dataset][1]
                    result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                    resulf_df = pd.read_csv(result_file_path)
                    results = resulf_df['total'].tolist()
                    assert len(results) == 5
                    mean,std = extract_saved_result(results[budget-1])
                    mean = round(mean*100,2)
                    mean = "{:.2f}".format(mean)

                    if adversarial == best:
                        row += f" &\\textbf{{{mean}}}"
                    elif adversarial == second:
                        row += f" &\\underline{{{mean}}}"
                    else:
                        row += f" & {mean}"
            row += f"& {round(adversarial_rank[adversarial],2)}\\\\"

            output += f"{row}"
        output += "\\hline"
    output += f"""
    \\end{{tabular}}
    }}
    \\end{{table*}}
    """

    return output


def generate_complete_table(config_setting,no_degree = False,adversarials_list = ['nettack'], models_list= ["GCN"],datasets_list = ['cora']):
    r"""
    Generate table in format as follow:

    \\begin{table}[ht]
    \\centering
    \\caption{}
    \\small
    \\renewcommand\\arraystretch{1.2}
    \\setlength{\\tabcolsep}{3pt}
    \\label{tab:missclass-table}
        \\resizebox{\\linewidth}{!}{ 
        \\begin{tabular}{ c| cc | ccccc | ccccc | ccccc | c  }
        \\toprule
           \\multicolumn{3}{c}{Dataset} & \\multicolumn{5}{c}{Cora} & \\multicolumn{5}{c}{Citeseer} & \\multicolumn{5}{c}{Pubmed} \\\\
           \\hline
           && $\\Delta \\rightarrow$ & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5& Avg. Rank \\\\
    """
    columns = 'c | cc | ' + ' | '.join(['c'* 5] * len(datasets_list)) + ' | c'
    datasets_row = ' & '.join([f'\\multicolumn{{5}}{{c}}{{{dataset}}}' for dataset in datasets_list])
    budget_row = ' & '.join(([' & '.join(['1','2','3','4','5'])] * len(datasets_list) ))

    table_head = f"&\\multicolumn{{1}}{{c}}{{Dataset}}"
    for dataset in datasets_list:
        table_head += f"& \\multicolumn{{{len(models_list)}}}{{c}}{{{dataset}}}"

    table_head += "& \\\\ \\midrule"
    
    output = f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{}}
    \\small
    \\renewcommand\\arraystretch{{1.2}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\label{{tab:missclass-table}}
        \\resizebox{{\\linewidth}}{{!}}{{ 
                \\begin{{tabular}}{{ {columns} }}
                \\toprule
                \\multicolumn{{3}}{{c}}{{Dataset}} & {datasets_row} \\\\
                \\hline
                && $\\Delta \\rightarrow$ & {budget_row} & Avg. Rank\\\\
                \\cline{{2-19}}


    """
    overal_ranking = {}
    # print(output)
    for setting in ["evasion","poision"]:
        adversarial_ranking_all = {}
        best_second_record_all = {}

        for budget in [1,2,3,4,5]:
            
            best_second_record_budget = {}
            for model in models_list:
                if not model in adversarial_ranking_all:
                    adversarial_ranking_all[model]= {}

                for dataset in datasets_list:
                    adversarial_performance = []
                    for adversarial in adversarials_list:
                        try:

                            result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                            resulf_df = pd.read_csv(result_file_path)
                            results = resulf_df['total'].tolist()
                            assert len(results) == 5
                            mean,std = extract_saved_result(results[budget-1])
                            adversarial_performance.append((adversarial,mean))
                        except Exception as e:
                            print(e)
                            adversarial_performance.append((adversarial,0))
                        
                    adversarial_sort = [x[0] for x in sorted(adversarial_performance,reverse=True, key=lambda item: item[1])]
                    for idx,sort_adv in enumerate(adversarial_sort):
                        if not sort_adv in adversarial_ranking_all[model]:
                            adversarial_ranking_all[model][sort_adv]= []
                        
                        if not sort_adv in overal_ranking:
                            overal_ranking[sort_adv] = []

                        adversarial_ranking_all[model][sort_adv].append(idx + 1)
                        overal_ranking[sort_adv].append(idx + 1)

                    
                    if not model in best_second_record_budget:
                        best_second_record_budget[model] = {}
                    
                    best_second_record_budget[model][dataset] = adversarial_sort[:2]

                
            best_second_record_all[budget] = best_second_record_budget

        
        for model in adversarial_ranking_all.keys():
            adversarial_rank = adversarial_ranking_all[model]
            for adversarial in adversarial_rank.keys():
                # print(model,adversarial,adversarial_ranking_all[model][adversarial])
                adversarial_ranking_all[model][adversarial] = stat.mean(adversarial_ranking_all[model][adversarial])
                

        multi_row = len(adversarials_list)
        for indx_model, model in enumerate(models_list): 

            for idx_adver,adversarial in enumerate(adversarials_list):
                if idx_adver == 0:
                    if indx_model == 0:
                        multi_row = len(models_list) * len(adversarials_list)
                        row = f"""
                        \\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{setting_dict[setting]}}}}}}}"""
                    else:
                        row = f""""""
                    multi_row = len(adversarials_list)
                    row += f"&\\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{model}}}}}}}"
                else:
                    row = """
                        &"""
                row += f"& {adversarial_name[adversarial]}"

                for dataset in datasets_list:
                    result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                    for budget in [1,2,3,4,5]:
                        try:

                            best = best_second_record_all[budget][model][dataset][0]
                            second = best_second_record_all[budget][model][dataset][1]
                            resulf_df = pd.read_csv(result_file_path)
                            results = resulf_df['total'].tolist()
                            assert len(results) == 5
                            mean,std = extract_saved_result(results[budget-1])
                            mean = round(mean*100,2)
                            mean = "{:.2f}".format(mean)
                            std = "{:.2f}".format(std)

                            if adversarial == best:
                                row += f" &\\textbf{{{mean} $\\pm$ {std}}}"
                            elif adversarial == second and len(adversarials_list) > 2:
                                row += f" &\\underline{{{mean} $\\pm$ {std}}}"
                            else:
                                row += f" & {mean} $\\pm$ {std}"
                        except Exception as e:
                            print(e)
                            row += f" & XX.XX $\\pm$ XX"
                rank = "{:.2f}".format(adversarial_ranking_all[model][adversarial])
                row += f"& {rank}\\\\"

                output += f"{row}"
            output += f"""
                        \\cline{{2-19}}
                    """
        output += "\\hline"
    output += f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\end{{table}}
    """

    for adversarial in overal_ranking.keys():
        overal_ranking[adversarial] = stat.mean(overal_ranking[adversarial])
    

    return output,overal_ranking

def generate_difference_with_our_evaluation_table(config_setting,no_degree = False,adversarials_list = ['nettack'], models_list= ["GCN"],datasets_list = ['cora']):
    r"""
    Generate table in format as follow:

    \\begin{table}[ht]
    \\centering
    \\caption{}
    \\small
    \\renewcommand\\arraystretch{1.1}
    \\setlength{\\tabcolsep}{10pt}
    \\label{tab:missclass-table}
        \\resizebox{\\linewidth}{!}{ 
        \\begin{tabular}{ c| cc | ccccc | ccccc | ccccc | c  }
        \\toprule
           \\multicolumn{3}{c}{Dataset} & \\multicolumn{5}{c}{Cora} & \\multicolumn{5}{c}{Citeseer} & \\multicolumn{5}{c}{Pubmed} \\\\
           \\hline
           && $\\Delta \\rightarrow$ & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5& Avg. Rank \\\\
    """
    columns = 'c | cc | ' + ' | '.join(['c'* 5] * len(datasets_list)) + ' | c'
    datasets_row = ' & '.join([f'\\multicolumn{{5}}{{c}}{{{dataset}}}' for dataset in datasets_list])
    budget_row = ' & '.join(([' & '.join(['1','2','3','4','5'])] * len(datasets_list) ))

    table_head = f"&\\multicolumn{{1}}{{c}}{{Dataset}}"
    for dataset in datasets_list:
        table_head += f"& \\multicolumn{{{len(models_list)}}}{{c}}{{{dataset}}}"

    table_head += "& \\\\ \\midrule"
    
    output = f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{}}
    \\small
    \\renewcommand\\arraystretch{{1.2}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\label{{tab:missclass-table}}
        \\resizebox{{\\linewidth}}{{!}}{{ 
                \\begin{{tabular}}{{ {columns} }}
                \\toprule
                \\multicolumn{{3}}{{c}}{{Dataset}} & {datasets_row} \\\\
                \\hline
                && $\\Delta \\rightarrow$ & {budget_row} & Avg. Rank \\\\
                \\cline{{2-19}}


    """
    # print(output)
    difference_list = []
    for setting in ["evasion","poision"]:
        our_adversarial_ranking_all = {}
        compared_adversarial_ranking_all = {}
        
        for budget in [1,2,3,4,5]:
            
            for model in models_list:
                if not model in our_adversarial_ranking_all:
                    our_adversarial_ranking_all[model]= {}
                    compared_adversarial_ranking_all[model] = {}

                for dataset in datasets_list:
                    our_adversarial_performance = []
                    compared_adversarial_performance = []
                    for adversarial in adversarials_list:
                        try:

                            our_result_file_path = f"{RESULT_PATH}/best_config/{model}/{dataset}/{adversarial}_{setting}.csv"
                            our_resulf_df = pd.read_csv(our_result_file_path)
                            our_results = our_resulf_df['total'].tolist()
                            assert len(our_results) == 5
                            our_mean,std = extract_saved_result(our_results[budget-1])
                            our_adversarial_performance.append((adversarial,our_mean))

                            if config_setting == "default":
                                compared_results_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                            elif config_setting == "no_degree":
                                compared_results_file_path = f"{RESULT_PATH}/ablation/{config_setting}/best_config/{model}/{dataset}/{adversarial}_{setting}.csv"
                            compare_resulf_df = pd.read_csv(compared_results_file_path)
                            compare_results = compare_resulf_df['total'].tolist()
                            assert len(compare_results) == 5
                            compare_mean,std = extract_saved_result(compare_results[budget-1])
                            compared_adversarial_performance.append((adversarial,compare_mean))

                            
                           
                        except Exception as e:
                            print(e)
                            our_adversarial_performance.append((adversarial,0))
                        
                    our_adversarial_sort = [x[0] for x in sorted(our_adversarial_performance,reverse=True, key=lambda item: item[1])]
                    for idx,sort_adv in enumerate(our_adversarial_sort):
                        if not sort_adv in our_adversarial_ranking_all[model]:
                            our_adversarial_ranking_all[model][sort_adv]= []
                        our_adversarial_ranking_all[model][sort_adv].append(idx + 1)

                    compare_adversarial_sort = [x[0] for x in sorted(compared_adversarial_performance,reverse=True, key=lambda item: item[1])]
                    for idx,sort_adv in enumerate(compare_adversarial_sort):
                        if not sort_adv in compared_adversarial_ranking_all[model]:
                            compared_adversarial_ranking_all[model][sort_adv]= []
                        compared_adversarial_ranking_all[model][sort_adv].append(idx + 1)
                    

        
        for model in our_adversarial_ranking_all.keys():
            adversarial_rank = our_adversarial_ranking_all[model]
            for adversarial in adversarial_rank.keys():
                our_adversarial_ranking_all[model][adversarial] = stat.mean(our_adversarial_ranking_all[model][adversarial])

        for model in compared_adversarial_ranking_all.keys():
            adversarial_rank = compared_adversarial_ranking_all[model]
            for adversarial in adversarial_rank.keys():
                compared_adversarial_ranking_all[model][adversarial] = stat.mean(compared_adversarial_ranking_all[model][adversarial])
                
        multi_row = len(adversarials_list)
        for indx_model, model in enumerate(models_list): 

            for idx_adver,adversarial in enumerate(adversarials_list):
                if idx_adver == 0:
                    if indx_model == 0:
                        multi_row = len(models_list) * len(adversarials_list)
                        row = f"""
                        \\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{setting_dict[setting]}}}}}}}"""
                    else:
                        row = f""""""
                    multi_row = len(adversarials_list)
                    row += f"&\\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{model}}}}}}}"
                else:
                    row = """
                        &"""
                row += f"& {adversarial_name[adversarial]}"

                for dataset in datasets_list:
                    our_result_file_path = f"{RESULT_PATH}/best_config/{model}/{dataset}/{adversarial}_{setting}.csv"
                    if config_setting == "default":
                        compared_results_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                    elif config_setting == "no_degree":
                        compared_results_file_path = f"{RESULT_PATH}/ablation/{config_setting}/best_config/{model}/{dataset}/{adversarial}_{setting}.csv"
                    for budget in [1,2,3,4,5]:
                        # try:
                        our_resulf_df = pd.read_csv(our_result_file_path)
                        our_results = our_resulf_df['total'].tolist()
                        assert len(our_results) == 5
                        our_mean,our_std = extract_saved_result(our_results[budget-1])

                        compared_resulf_df = pd.read_csv(compared_results_file_path)
                        compared_results = compared_resulf_df['total'].tolist()
                        # print(compared_results_file_path)
                        # print(compared_results)
                        assert len(compared_results) == 5
                        compared_mean,compared_std = extract_saved_result(compared_results[budget-1])

                        difference = round((our_mean-compared_mean)*100,2)
                        difference_list.append(difference)
                        if difference == 0:
                            difference = "{:.2f}".format(difference)
                        elif difference > 0:
                            difference = "\positive{{{:.2f}}}".format(difference)
                        elif difference < 0:
                            difference = "\\negative{{{:.2f}}}".format(difference)
                        row += f" & {difference}"
                        # except Exception as e:
                            # print(e)
                            # row += f" & XX.XX $\\pm$ XX"

                        
                # row += "\\\\"
                rank = our_adversarial_ranking_all[model][adversarial] - compared_adversarial_ranking_all[model][adversarial]
                if rank == 0:
                    rank = "{:.2f}".format(rank)
                elif rank > 0:
                    rank = "\positive{{{:.2f}}}".format(rank)
                elif rank < 0:
                    rank = "\\negative{{{:.2f}}}".format(rank)
                row += f"& {rank}\\\\"


                output += f"{row}"
            output += f"""
                        \\cline{{2-19}}
                    """
        output += "\\hline"
    output += f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\end{{table}}
    """
    
    return output,stat.mean(difference_list)


def generate_complete_table_for_defense(config_setting,no_degree = False, models_list= ["GCN"], dataset = "cora"):
    r"""
    Generate table in format as follow:

    \\begin{table}[ht]
    \\centering
    \\caption{}
    \\small
    \\renewcommand\\arraystretch{1.2}
    \\setlength{\\tabcolsep}{3pt}
    \\label{tab:missclass-table}
        \\resizebox{\\linewidth}{!}{ 
        \\begin{tabular}{ c| cc | ccccc | ccccc | ccccc | c  }
        \\toprule
           \\multicolumn{3}{c}{Dataset} & \\multicolumn{5}{c}{Cora} & \\multicolumn{5}{c}{Citeseer} & \\multicolumn{5}{c}{Pubmed} \\\\
           \\hline
           && $\\Delta \\rightarrow$ & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5& Avg. Rank \\\\
    """
    columns = 'c | cc | ccccc | c'
    budget_row = ' & '.join(['1','2','3','4','5'])

    
    output = f"""
    \\begin{{table}}[ht]
    \\centering
    \\caption{{}}
    \\small
    \\renewcommand\\arraystretch{{1.2}}
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\label{{tab:missclass-table}}
        \\resizebox{{\\linewidth}}{{!}}{{ 
                \\begin{{tabular}}{{ {columns} }}
                \\toprule
                \\hline
                && $\\Delta \\rightarrow$ & {budget_row} & Avg. Rank\\\\
                \\cline{{2-9}}


    """
    adversarials_list = defense_adversarial[dataset]
    # print(output)
    for setting in ["evasion","poision"]:
        defense_ranking_all = {}
        best_second_record_all = {}

        for budget in [1,2,3,4,5]:
            
            best_second_record_budget = {}
            for adversarial in adversarials_list:
            
                if not adversarial in defense_ranking_all:
                    defense_ranking_all[adversarial]= {}

                adversarial_performance = []
                # for adversarial in adversarials_list:
                for model in models_list:
                    try:
                        result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                        resulf_df = pd.read_csv(result_file_path)
                        results = resulf_df['total'].tolist()
                        assert len(results) == 5
                        mean,std = extract_saved_result(results[budget-1])
                        adversarial_performance.append((model,mean))
                    except Exception as e:
                        print(e)
                        adversarial_performance.append((model,0))
                    
                defense_sort = [x[0] for x in sorted(adversarial_performance, key=lambda item: item[1])]
                for idx,sort_defense in enumerate(defense_sort):
                    if not sort_defense in defense_ranking_all[adversarial]:
                        defense_ranking_all[adversarial][sort_defense]= []
                    
                    

                    defense_ranking_all[adversarial][sort_defense].append(idx + 1)

                
                # if not adversarial in best_second_record_budget:
                #     best_second_record_budget[adversarial] = {}
                
                best_second_record_budget[adversarial] = defense_sort[:2]

                
            best_second_record_all[budget] = best_second_record_budget

        
        for adversarial in defense_ranking_all.keys():
            model_rank = defense_ranking_all[adversarial]
            for model in model_rank.keys():
                defense_ranking_all[adversarial][model] = stat.mean(defense_ranking_all[adversarial][model])
                

        # multi_row = len(models_list)
        for idx_adver,adversarial in enumerate(adversarials_list):

            for indx_model, model in enumerate(models_list): 
            # for idx_adver,adversarial in enumerate(adversarials_list):
                if indx_model == 0:
                    if idx_adver == 0:
                        multi_row = len(models_list) * len(adversarials_list)
                        row = f"""
                        \\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{setting_dict[setting]}}}}}}}"""
                    else:
                        row = f""""""
                    multi_row = len(models_list)
                    row += f"&\\multirow{{{multi_row}}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{adversarial_name[adversarial]}}}}}}}"
                else:
                    row = """
                        &"""
                model_name = model.replace("_","-")
                row += f"& {model_name}"

                result_file_path = f"{RESULT_PATH}/{config_setting}/{model}/{dataset}/{adversarial}_{setting}.csv"
                for budget in [1,2,3,4,5]:
                    try:

                        best = best_second_record_all[budget][adversarial][0]
                        second = best_second_record_all[budget][adversarial][1]
                        resulf_df = pd.read_csv(result_file_path)
                        results = resulf_df['total'].tolist()
                        assert len(results) == 5
                        mean,std = extract_saved_result(results[budget-1])
                        mean = round(mean*100,2)
                        mean = "{:.2f}".format(mean)
                        std = "{:.2f}".format(std)

                        if model == best:
                            row += f" &\\textbf{{{mean} $\\pm$ {std}}}"
                        elif model == second and len(models_list) > 2:
                            row += f" &\\underline{{{mean} $\\pm$ {std}}}"
                        else:
                            row += f" & {mean} $\\pm$ {std}"
                    except Exception as e:
                        print(e)
                        row += f" & XX.XX $\\pm$ XX"
                rank = "{:.2f}".format(defense_ranking_all[adversarial][model])
                row += f"& {rank}\\\\"

                output += f"{row}"
            output += f"""
                        \\cline{{2-9}}
                    """
        output += "\\hline"
    output += f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\end{{table}}
    """

    # for adversarial in overal_ranking.keys():
    #     overal_ranking[adversarial] = stat.mean(overal_ranking[adversarial])
    

    return output

if __name__ == "__main__":
    adversarials_list = ['random_degree_attack','fga','nettack','pgdattack','prbcdattack','sga','gottack']
    # adversarials_list = ['random_degree_attack','rnd']

    # adversarials_list = ['fga','nettack','pgdattack','prbcdattack','sga']
    models_list = ['GCN',"GIN","GSAGE","PNA"]
    defense_models = ['GCN_GARNET',"GNNGuard","GRAND"]
    # datasets_list = ['cora','citeseer','pubmed']
    datasets_list = ['cora']

    output,overal_ranking = generate_complete_table("best_config",adversarials_list=adversarials_list,models_list=models_list,datasets_list=datasets_list)
    print(output)
    print(overal_ranking)
    
    # difference
    # adversarials_list = ['fga','nettack','pgdattack','prbcdattack','sga','gottack']
    # models_list = ['GCN',"GIN","GSAGE","PNA"]
    # datasets_list = ['cora','citeseer','pubmed']
    # output,mean_diff = generate_difference_with_our_evaluation_table("default",adversarials_list=adversarials_list,models_list=models_list,datasets_list=datasets_list)
    # print(output)
    # print(mean_diff)
    
    # defense_models = ['GCN_Jaccard','GCN_GARNET',"GNNGuard","GRAND","ElasticGNN","RobustGCN"]
    # output = generate_complete_table_for_defense("best_config",models_list=defense_models + models_list,dataset="pubmed")
    # print(output)
