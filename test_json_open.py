import gzip
import json

with gzip.open('/raid/datasets/steervla/simplified_reasoning/simlingo/lb1_split/routes_training/ControlLoss/Town04_Rep0_Town04_Scenario1_55_route0_01_11_06_49_16/reasoning/0057.json.gz', 'rb') as f:
    data = json.load(f)

breakpoint()
print(data)