import sys
import os
import statistics as stats
import pandas as pd
import timeit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import AdversarialAssessment
from utility.config import args
from static import *

def save_results(results:dict,result_path:str):
    
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(
            columns=results.keys())
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append(results, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def get_stat(results_list):
    return (stats.mean(results_list),stats.stdev(results_list))



if __name__ == "__main__":
    
    # Need to set up path here
    adaptive = ""
    classifier_config_version = "best_config" #@TODO: remove it?
    if args.adaptive_surrogate == args.model and args.adversarial == PRBCD_NAME:
        adaptive = "_adaptive" #@TODO: Check if AdversarialAssessment has this ability


    if args.purification is not None: #@TODO: Make this look nicer
        if not os.path.exists("{}{}/{}_{}/{}".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset)):
            os.makedirs("{}{}/{}_{}/{}".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset))

        poision_path = "{}{}/{}_{}/{}/{}{}_poision.csv".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset,args.adversarial,adaptive)
        evasion_path = "{}{}/{}_{}/{}/{}{}_evasion.csv".format(PATH_RESULT,classifier_config_version,args.model,args.purification,args.dataset,args.adversarial,adaptive)
    else:
        if not os.path.exists("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset)):
            os.makedirs("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset))

        poision_path = "{}{}/{}/{}/{}{}_poision.csv".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial,adaptive)
        evasion_path = "{}{}/{}/{}/{}{}_evasion.csv".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial,adaptive)

    
    evalution_pipeline = AdversarialAssessment(
        model= args.model,
        dataset=args.dataset,
        adversarial=args.adversarial,
        num_runs=args.num_run,
        num_split=args.num_split,
        base_seed=args.seed,
        purification=args.purification, #None by default
        device=args.device
    )
    for budget in [1,2,3,4,5]:
        evasion_dict,poision_dict = evalution_pipeline.evaluate(
            evasion=True,
            poision=True,
            budgets_list=[budget]
        ) #@TODO consider make this return dict instead

        save_results(evasion_dict,evasion_path)
        save_results(poision_dict,poision_path)
