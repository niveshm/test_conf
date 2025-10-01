import json

dataset = "icews14"
rule_file = "150925115904_r[1,2,3]_n200_exp_s12_rules.json"

rule_path = f"rules/{dataset}/{rule_file}"

with open(rule_path, "r") as f:
    rules = json.load(f)



breakpoint()