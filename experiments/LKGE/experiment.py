import subprocess
import sys


seeds = ["11"]
inits = ["0"]
gains = ["1"]
datasets = ["ENTITY","FACT","RELATION","HYBRID","graph_equal","graph_higher","graph_lower"]
ics = ["EMR","EWC","LKGE"]

for seed in seeds:
        for init in inits:
                for ic in ics:
                        for dataset in datasets:
                                #fixed_args = [
                                #"-dataset", "ENTITY", "-reuse_0"
                                #]
                                #command = [sys.executable, "main.py"] + fixed_args + ["-seed", seed, "-init",init, "-gain",gain, "-lifelong_name",ic]
                                command = [sys.executable, "main.py"] +  ["-dataset",dataset,"-seed", seed, "-init",init, "-gain",gain, "-lifelong_name",ic]
                                subprocess.run(command)

seeds = ["11"]
inits = ["0"]
gains = ["1"]
datasets = ['PS-CKGE']
ics = ["EMR","EWC","LKGE"]

for seed in seeds:
        for init in inits:
                for gain in gains:
                        for ic in ics:
                                for dataset in datasets:
                                        #fixed_args = [
                                        #"-dataset", "ENTITY", "-reuse_0"
                                        #]
                                        #command = [sys.executable, "main.py"] + fixed_args + ["-seed", seed, "-init",init, "-gain",gain, "-lifelong_name",ic]
                                        command = [sys.executable, "main.py"] +  ["-dataset",dataset,"-seed", seed, "-init",init, "-gain",gain, "-lifelong_name",ic, "-snapshot_num","3"]
                                        subprocess.run(command)



seeds = ["11"]
datasets = ["ENTITY","FACT","RELATION","HYBRID","graph_equal","graph_higher","graph_lower"]
ics = ["retraining"]

for seed in seeds:
        for ic in ics:
                for dataset in datasets:
                        for lr in ["0.001","0.0005","0.00005"]:
                        
                                command = [sys.executable, "main.py"] +  ["-dataset",dataset,"-seed", seed, "-learning_rate",lr,"-skip_previous","True", "-lifelong_name",ic]
                                subprocess.run(command)

seeds = ["11"]
datasets = ['PS-CKGE']
ics = ["retraining"]

for seed in seeds:
        for ic in ics:
                for dataset in datasets:
                        for lr in ["0.001","0.0005","0.00005"]:
                        
                                command = [sys.executable, "main.py"] +  ["-dataset",dataset,"-seed", seed, "-learning_rate",lr,  "-skip_previous","True", "-lifelong_name",ic, "-snapshot_num","3"]
                                subprocess.run(command)





# seeds = ["11"]
# datasets = ["RELATION"]
# ics = ["LKGE"]

# for regular_weight in ['1']:
#         for reconstruct_weight in ['1']:
#                 for dataset in datasets:
                        
#                         command = [sys.executable, "main.py"] +  ["-dataset",dataset,"-seed", str(33), "-skip_previous","True", "-lifelong_name","LKGE", "-regular_weight",regular_weight, "-reconstruct_weight", reconstruct_weight]
#                         subprocess.run(command)
