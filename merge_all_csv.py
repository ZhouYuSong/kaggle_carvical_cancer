import pandas as pd
import numpy as np
import os

def merge_all_csv(csv_folder, final_csv_name):
    csv_names = os.listdir(csv_folder)
    sample_csv = pd.read_csv(os.path.join(csv_folder, csv_names[0]), index_col=0) 
    final_data = np.zeros(sample_csv.values.shape)
    final_index = sample_csv.index
    final_columns = sample_csv.columns
    lines_size = final_data.shape[0]
    print lines_size
    step = 200
    line_indx = 0
    for start in range(0, lines_size, step):
        data_list = [pd.read_csv(os.path.join(csv_folder, csv_file), index_col=0, nrows=step, skiprows=start) for csv_file in csv_names]		
        lines_num = data_list[0].shape[0]
	for line_idx in range(lines_num):
	    type1_list = []
	    type2_list = []
	    type3_list = []
	    for idx, data in enumerate(data_list):
		if data.iloc[line_idx, 0] > data.iloc[line_idx, 1] and data.iloc[line_idx, 0] > data.iloc[line_idx, 2]:
		    type1_list.append(idx)
	  	elif data.iloc[line_idx, 1] > data.iloc[line_idx, 0] and data.iloc[line_idx, 1] > data.iloc[line_idx, 2]:
		    type2_list.append(idx)
		else:
		    type3_list.append(idx)
	    res_list = []
	    if len(type2_list) >= len(type1_list) and len(type2_list) >= len(type3_list):
		res_list = type2_list
	    elif len(type3_list) >= len(type2_list) and len(type3_list) >= len(type1_list):
		res_list = type3_list 
	    else:
		res_list = type1_list
	
	    for idx in res_list:
		final_data[start + line_idx] += data_list[idx].iloc[line_idx]
	    if len(res_list) == 0:
	        print start+line_idx, lines_num, len(type1_list), len(type2_list), len(type3_list)
	    final_data[start + line_idx] /= len(res_list)    	 
    df = pd.DataFrame(final_data, index=final_index, columns=final_columns)
    df.to_csv("final.csv")

if __name__ == "__main__":
    merge_all_csv("submodels", "falnal.csv")
