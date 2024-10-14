

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.config import args
from evaluation import AdversarialAssessment


if __name__ == "__main__":
    args.dataset = "cora"
    args.model= "GCN"
    args.adversarial= "fga"
    assesment = AdversarialAssessment(model = args.model,dataset = args.dataset,adversarial= args.adversarial,
                          device=args.device)
    
    assesment.evaluate(budgets_list=[1])