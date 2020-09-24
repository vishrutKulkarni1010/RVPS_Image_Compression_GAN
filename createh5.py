import glob
import pandas as pd
a=[]
for name in glob.glob('train/*'): 
    a.append(name[6:])
initial_data = {}
df = pd.DataFrame(initial_data, columns = []) 
df["path"]=a
print(df)
import pandas as pd
hf = pd.HDFStore('mydata.h5')
hf.put('df',df)
hf.close()
ab = pd.HDFStore('mydata.h5',mode='r')
print(ab.keys())
df1 = ab.get('df')
print(df1)
