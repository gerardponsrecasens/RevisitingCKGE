import subprocess
import sys




seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "graph_higher"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init]
                                subprocess.run(command)

seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "graph_lower"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init]
                                subprocess.run(command)

seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "PS-CKGE"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init, "-snapshot_num","3"]
                                subprocess.run(command)

seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "FACT"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init]
                                subprocess.run(command)


seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "RELATION"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init]
                                subprocess.run(command)


seeds = ["55"]
inits = ["0"]
rel_inits = ["0"]
ics = ["incDE"]

for seed in seeds:
        for init in inits:
                for rel_init in rel_inits:
                        for ic in ics:
                                fixed_args = [
                                    "-dataset", "HYBRID"
                                ]
                                command = [sys.executable, "main.py"] + fixed_args + ["-init",init]
                                subprocess.run(command)

