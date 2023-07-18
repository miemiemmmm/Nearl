import numpy as np 

def pdbbind_to_csv(inputfile, outputfile): 
  with open(inputfile, "r") as file1: 
    entries = [i for i in file1.read().strip("\n").split("\n") if "#" not in i]
    entries = [i.split()[:4]+[i.split()[-1].replace(",","_")] for i in entries]
    entries = np.array(entries)
    print(f"Found {len(entries)} entries")
    final_str = "pdbcode,resolution,year,pK,comment\n"
    for entry in entries: 
      final_str += f"{entry[0]},{entry[1]},{entry[2]},{entry[3]},{entry[4]}\n"
  with open(outputfile, "w") as file1: 
    file1.write(final_str)
    

pdbbind_to_csv(
  "INDEX_general_PL_data.2020", 
  "PDBBind_general_v2020.csv"
)
pdbbind_to_csv(
  "INDEX_refined_data.2020",
  "PDBBind_refined_v2020.csv"
)

