import json

data=[]
base_path="result_base/"
for i in range(1,20):
    j=i*24*7
    fp=open(base_path+"after{:04}.result.json".format(j))
    obj=json.load(fp)
    for k,v in obj["all"].items():
        data.append([k,"after","{:02d}".format(i),i*7]+v["y_prob"])
base_path="result_base/"
for i in range(0,20):
    j=i*24*7
    fp=open(base_path+"prev{:03}.result.json".format(j))
    obj=json.load(fp)
    for k,v in obj["all"].items():
        data.append([k,"prev","{:02d}".format(i),i*7]+v["y_prob"])

h=["ID","prev/after","step","day","non ROP", "APROP", "Type1 ROP"]
out_filename="prev_after_summary.csv"
fp=open(out_filename,"w")
fp.write(",".join(h))
fp.write("\n")
for el in sorted(data):
    fp.write(",".join(map(str,el)))
    fp.write("\n")
