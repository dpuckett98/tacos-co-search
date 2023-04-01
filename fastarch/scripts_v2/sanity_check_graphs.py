import matplotlib.pyplot as plt

# params
file_name = "sanity_check_full_v2.txt"
param_name = "c_w"
metric_name = "dram_accesses"

# load file into array
data = []
f = open(file_name)
lines = f.readlines()
for idx, line in enumerate(lines):
	if idx == 0:
		continue
	
	line = line.replace(',', '')
	line = line.replace(' ', '\t')
	line = line.replace('\n', '')
	l = line.split('\t')
	#print(l)
	
	curr = {}
	curr["cycles"] = float(l[0])
	curr["dram_accesses"] = float(l[1])
	curr["A_rows"] = float(l[3])
	curr["B_cols"] = float(l[4])
	curr["A_cols_B_rows"] = float(l[5])
	curr["num_PEs"] = float(l[15])
	curr["num_RFs"] = float(l[16])
	curr["size_RFs"] = float(l[17])
	curr["off_chip_bandwidth"] = float(l[18])
	curr["on_chip_bandwidth"] = float(l[19])
	curr["buffer_size"] = float(l[20])
	curr["dataflow"] = l[21]
	curr["t_a"] = float(l[22])
	curr["t_b"] = float(l[23])
	curr["t_w"] = float(l[24])
	curr["c_a"] = float(l[25])
	curr["c_b"] = float(l[26])
	curr["c_w"] = float(l[27])
	#print(curr)
	data.append(curr)

print(len(data))

# plot
y_values = [c[metric_name] for c in data]
#x_values = [c["t_a"] * c["c_b"] + c["c_w"] * c["c_a"] + c["c_w"] * c["c_b"] for c in data]
x_values = [c[param_name] for c in data]
plt.plot(x_values, y_values, '.')
plt.xlabel(param_name)
plt.ylabel(metric_name)
plt.yscale("log")
plt.show()