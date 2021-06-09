# # Analysis
# time = [] # float: 21910 elements, value: (0 - 1739.4) // int: 1314 elements, value: (0 - 1739)
# frontal_axis = [] # 175 elements
# vertical_axis = [] # 169 elements
# lateral_axis = [] # 205 elements
# sensor_id = [] # 4 elements: 1 2 3 4
# rssi = [] # 68 elements
# phase = [] # 3814 elements (0 - 6.2817)
# f = [] # 12 elements: ['920.25', '920.75', '921.25', '921.75', '922.25', '922.75', '923.25', '923.75', '924.25', '924.75', '925.25', '925.75']

# fp = open("test2.csv", "r")
# fp.readline()
# for line in fp.readlines():
#     time.append(int(float(line.split(",")[0])))
#     frontal_axis.append(line.split(",")[1])
#     vertical_axis.append(line.split(",")[2])
#     lateral_axis.append(line.split(",")[3])
#     sensor_id.append(line.split(",")[4])
#     rssi.append(line.split(",")[5])
#     phase.append(line.split(",")[6])
#     f.append(line.split(",")[7])
# fp.close()
# list_set = set(f)
# unique_time = list(list_set)
# unique_time = sorted(unique_time)
# print(unique_time)
# print(len(unique_time))

# # Make dataset
# import glob

# w = open("test2.csv","a")

# for file in glob.glob("Datasets_Healthy_Older_People/*/*"):
#     f = open(file,"r")
#     for line in f.readlines():
#         w.write(line)
#     f.close()
# w.close()