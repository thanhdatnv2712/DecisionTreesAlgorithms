import glob

w = open("test2.csv","a")

for file in glob.glob("Datasets_Healthy_Older_People/*/*"):
    f = open(file,"r")
    for line in f.readlines():
        w.write(line)
    f.close()
w.close()