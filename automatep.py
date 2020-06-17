
#def convert_k_m(miles):
 #   km= 1.6* miles
  #  print("km:")
   # print(km)
    #if km<8:
     #   return "is small"
    #else:
     #   return "is large"
#a=convert_k_m(5)
#b=convert_k_m(20)
#print(a)
#print(b)


#from pandas import DataFrame
# from matplotlib import pyplot as plt
#age=[1,2,3,4,5,]
#food=["a","b","c","d","e"]
#plt.plot(age,food,)
#plt.show()

#from matplotlib import pyplot as plt
#population=[1.0,1.262,1.650]
#year=[1800,1850,1900]

#plt.fill_between(year,population,0,color="green")
#plt.xlabel("Year")
#plt.ylabel("Population")
#plt.title("World Population Projections")
#plt.show()

#import pandas as pd
#popo = pd.read_csv("path/to/popo.csv")
#print(popo)

def new(a,b):
    return a*b

print(new(4,8))

for i in [1,2,3,4,5]:
    print(i)
    for j in [1,2,3,4,5]:
        print(j)
        print(i+j)
    print(i)

l=[1,2,3]
l.insert(2,8)
print(l)

def square(a):
    """ Return the square of a """
    return a**2
print(square(4))

from itertools import compress
data=[1,2,3]
dat=[4,5,6]
print(list(compress(data,dat)))

def square(x):
    return x**2
for n in range(1,4):
    print(n,"squared is ",square(n))

c=2.1
e=4.2
print(c+e)

from decimal import Decimal as D
a= D("4.2")
b=D("2.1")
print(a+b)

from decimal import localcontext
a=D("1.3")
b=D("1.7")
with localcontext() as l:
    l.prec=3
    print(a/b)

x=1234.56789
print(round(x,3))
y=1234
print(bin(y))
print(format(y, "b"))

print(int("1001110101110",2))
print(complex(2,4))

import numpy
print(numpy.__version__)

import numpy as np
print(np.array([1,4,2,5,3]))
print(np.array([3.14,4,2,3]))
print(np.array([5,6,7,8], dtype="float32"))
print(np.array([9,10,11,12], dtype="float32"))
print(np.array([range(i,i+3) for i in [2,4,6]]))

print(np.ones((3,5),dtype="int"))
print(np.full((3,5),3.14))
print(np.arange(0,20,2))
print(np.linspace(0,1,5))
print(np.random.randint(0,10,(3,3)))
print(np.eye((3),dtype="int"))

import numpy as np
print(np.random.seed(0))
x1= np.random.randint(10,size=6)
x2= np.random.randint(10,size=(3,4))
x3= np.random.randint(10,size=(3,4,5))
print(x1)
print(x2)
print(x3)

print("x3 ndim: ",x3.ndim)
print("x3 shape: ",x3.shape)
print("x3 size: ",x3.size)
print("x3 dtype: ",x3.dtype)
print(x1[0])
print(x2[0,0])
print(x2[2,-1])

x=np.arange(10)
print(x)
print(x2)
print(x2[:2])
print(x2[:2,:3])
print(x2[:2,::3])
print(x2)
print(x2[:4,::-1])
print(x2[::-1,::-1])
print(x3)
print(x3[::-1,::-1,::-1])
print(x2[:1])
print(x2)
print(x2[:,0])
print(x2[0,:])
print(x2)
print(x2[:3,:1])
x2_sub_copy= x2[:2,:2].copy()
print(x2_sub_copy)
grid=np.arange(1,10).reshape((3,3))
gridnew=grid.copy().reshape(1,9)
print(gridnew)
print(grid)
print(grid[:,0].reshape(3,1))
x=np.array([1,2,3])
y=np.array([3,2,1])
print(np.concatenate([x,y]))
z=[99,99,99]
print(np.concatenate([x,z,y]))
print(np.concatenate([grid,grid]))
print(np.concatenate([grid,grid],axis=1))
x=np.array([1,2,3])
grid=np.array([[9,8,7],
               [6,5,4]])
print(np.vstack([grid,x]))
y= np.array([[99],
             [99]])
print(y)
print(np.hstack([grid,y]))
x=[1,2,3,99,99,3,2,1]
print(x)
x1,x2,x3= np.split(x, [3, 6])
print(x1,x2,x3)
x=np.array([1,2,3,99,99,3,2,1])
print(x)
print(x1,x2,x3)
grid=np.arange(16).reshape(4,4)
print(grid)
upper,lower=np.vsplit(grid,[2])
print(upper)
print(lower)
left, right= np.hsplit(grid,[2])
print(left)
print(right)

import numpy as np
np.random.seed(0)

def compute_recprocals(values):
    output= np.empty(len(values))
    for i in range(len(values)):
        output[i]=1.0/values[i]
    return output

values=np.random.randint(1,10,size=5)
print(compute_recprocals(values))

big_array= np.random.randint(1,100,size=10000)
print(big_array)

x=np.arange(4)
print("x+5: ", x+5)
print("x*2 ", x*2)
print(-(0.5*x+1)**2)
print(np.negative(x))
a=np.negative(x)
print(a)
print(np.abs(a))
x=np.array([3-4j])
print(np.absolute(x))
a=(np.linspace(0,3,3))
print(a)
print(np.pi)
theta= np.linspace(0,np.pi,3)
print("theta: ", theta)
print("sin(theta): ",np.sin(theta))
print("cos(theta): ",np.cos(theta))
print("tan(theta): ",np.tan(theta))
print(np.arcsin(1))
x=np.array([1,2,3])
print("e^x: ",np.exp(x))
print("2^x: ",np.exp2(x))
print(np.power(x,2))
print(np.empty(5))
print(np.zeros(10))
z=np.empty(8)
print(z)
x=np.arange(5)
print(np.power(2,x))
y=np.empty(5)
np.multiply(x,10,out=y)
print(y)
y=np.zeros(10)
np.power(2,x,out=y[::2])
print(y)
x=np.arange(1,6)
print(np.add.reduce(x))
print(np.multiply.reduce(x))
print(np.add.accumulate(x))
print(np.multiply.accumulate(x))
import numpy as np
l=np.random.random(100)
print(np.sum(l))
big_array=np.random.rand(1000000)
print(np.sum(big_array))
print(np.min(big_array))
print(np.max(big_array))
m=np.random.random((3,4))
print(m)
print(np.sum(m))
print(np.min(m,axis=0))
print(np.max(m,axis=0))
import pandas as pd
data=np.array([189,170,189,163,183,171,185,168,173])
name=np.array(["a","b","c","d","e","f","g","h","i"])
print(name)
print(data)
print("Mean height: ",np.mean(data))
print("Standard deviation: ", np.std(data))
print("Minimum height: ",np.min(data))
print("Maximum height:",np.max(data))
print("25th percentile: ",np.percentile(data,25))
print("75th percentile: ",np.percentile(data,75))

#import matplotlib.pyplot as plt
#import seaborn; seaborn.set()


#plt.scatter(name,data)
#plt.title("Heights Distribution of US presidents")
#plt.ylabel("Height cm")
#plt.xlabel("name")

#plt.show()
m= np.ones((2,3))
print(m)
a=np.arange(3)
print(a)
print(np.shape(m))
print(np.shape(a))
print(np.ndim(m))
print(m+a)
print(a[:,np.newaxis])  #transposing a
x=np.random.random((10,3))
print(x)
x_mean= np.mean(x, axis=0)
y=np.array([1,2,3])
print(np.mean(y,axis=0))
x_centered= x-x_mean
print(np.mean(x_centered,axis=0))
x=np.linspace(0,5,50)
y=np.linspace(0,5,50)[:,np.newaxis]
z=np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)
print(x)
print(y)
print(z)
#import matplotlib.pyplot as plt
#plt.imshow(z,origin="lower",extent=[0,5,0,5],cmap="viridis")
#plt.colorbar()
#plt.title("2 dimensional map")
#plt.show()

#a=np.arange(1,10).reshape(3,3)
#print(a)
#print(np.ndim(a))
#import matplotlib.pyplot as plt
#plt.imshow(a,origin="lower",extent=[0,10,0,10],cmap="viridis")
#plt.colorbar()
#plt.show()

x=np.array([1,2,3,4,5])
print(x<3)
print(x.dtype)
for y in (2*x)==(x**2):
    print(y)
print(np.greater_equal(x,3))
rng=np.random.RandomState(0)
x=rng.randint(10,size=(3,4))
print(x)
print(np.less(x,6))
print(np.count_nonzero(np.less(x,6)))
print(np.sum(x<6,axis=1))
print(np.sum((x>1)&(x<5)))
print(x<5)
print(x[x<5])
print(np.arange(5))
#rainy=inches>0 #construct a mask of all raininy days
#rainy=(inches>0)
#summer=(np.arange(365)-172<90)&(np.arange(365)-172>0)
#print(np.median(inches[summer]))

print(bool(0))
print(format(42,"b"))
x=np.arange(10)
print(x)
print((np.greater(x,4))&(np.less(x,8)),x.dtype)
rand=np.random.RandomState(42)
x=rand.randint(100,size=10)
print(x)
print([x[3],x[7],x[2]])
ind=[3,7,4]
print(x[ind])
ind=np.array([[3,7],
             [4,5]])
print(x[ind])
x=np.arange(12).reshape(3,4)
print(x)
print(x[:,1])
row=np.array([0,1,2])
col=np.array([2,1,3])
print(x[row,col]) #[0,2,1,1,2,3]
print(x[row[:,np.newaxis],col])
print(int("1010",2))
mask=np.array([1,0,1,0],dtype=bool)
print(mask)
print(x[row[:,np.newaxis],mask])
mean=[0,0]
cov=[[1,2],
     [2,5]]
x=rand.multivariate_normal(mean,cov,100)
print(x)
import matplotlib.pyplot as plt
#plt.scatter(x[:,0],x[:,1])
#plt.show()
# rand=np.random.randint(12, size=12).reshape(3,4)
# print(rand)
# import matplotlib.pyplot as plt
# plt.scatter(rand[:,0],rand[:,1])
#plt.imshow(rand,extent=[0,15,0,15],origin="lower",cmap="gray")
#plt.colorbar()
#plt.show()
#indices=np.random.choice(x.shape[0],20,replace=False)
#print(indices)
x=np.arange(10)
i=[2,1,8,4]
print(x)
x[i]=99
print(x)
x[i]-=10
print(x)
x=np.zeros(10)
x[[0,0]]=[4,6]
print(x)
i=[2,3,3,4,4,4]
x[i]+=1
print(x)
np.random.seed(42)
x=np.random.randn(100)
print(x)
bins=np.linspace(-5,5,20)
print(bins)
#plt.hist(x,bins,histtype="step")
#plt.show()
rand=np.random.RandomState(42)
x=rand.randint(0,10,(4,6))
a,b,c=np.split(x,[1,2])
print(a)
print(b)
print(c)
print(x)
#print(np.sort(x,axis=0))
#print(np.sort(x,axis=1))
#x=np.array([7,2,3,1,6,5,4])
#print(np.partition(x,3))
x=rand.rand(10,2)
print(x)
import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1],s=100,c="green",edgecolors="blue")
#plt.show()
a=np.arange(0,4).reshape(2,2)
print(a)
print((a[:,np.newaxis]))
print(a[np.newaxis,:])
print(x[:,np.newaxis])
dist_sq= np.sum((x[:,np.newaxis]-x[np.newaxis,:])**2,axis=-1)
differences= x[:,np.newaxis]-x[np.newaxis,:]
print(differences)
print(np.shape(differences))
print(dist_sq)
print(np.shape(dist_sq))
a=np.array([1,2,3])
print(np.shape(a))
b=np.random.randint(10,size=(3,4,5))
print(b)
print(np.diagonal(dist_sq))
print("aaaaaaaaaaa")
name=["Alice","Bob","cathy","doug"]
age=[25,45,37,19]
weight=[55.0,85.5,68.0,61.5]
print(list(compress(name,age)))
print(np.zeros(4,dtype=int))
data=np.zeros(4,dtype={"names":("name","age","weight"),"formats":("U10","i4","f8")})
print(data.dtype)
print(data)
data["name"]=name
data["age"]=age
data["weight"]=weight
print(data)
print(data["name"])
fam=("a","b","c")
no=(1,2,3)
ye=(1.0,2.0,3.0)
fam_Data=np.zeros(3,dtype={"names":("fam","no","ye"),"formats":("U10","i4","f8")})
print(fam_Data)
fam_Data["fam"]=fam
fam_Data["no"]=no
fam_Data["ye"]=ye
print(fam_Data)
print(data[3]["name"])
print(data[data["age"]<30]["name"])
print("THIS IS PANDAS SZNNNNNNNNNNNNNNNNNNNN")
import pandas
print(pandas.__version__)
import pandas as pd
data=[0.23,0.5,0.75,1.0]
#for i in list(enumerate(data,1)):
    #print(i)
data=pd.Series([0.25,0.5,0.75,1.0])
print(data)
print(data.values)
print(data.index)
data=pd.Series([0.25,0.5,0.75,1.0],index=["a","b","c","d"])
print(data)
print(data["a"])
population_dict={"California":38332521,"Texas":26448193,"New York":19651127,"Florida":19552860,"Illinois":12882135}
population=pd.Series(population_dict)
print(population)
print(population["California":"Illinois"])
area_dict={"California":423967,"Texas":695662,"New York":141297,"Florida":170312,"Illinois":149995}
area=pd.Series(area_dict)
print(area)
states=pd.DataFrame({"population":population,"area":area})
print(states)
print(states.index)
print(states.columns)
print(states["area"])
data=[{"a":i,"b":2*i} for i in range(3)]
print(pd.DataFrame(data))
data=pd.Series({"a":1,"b":2})
print(pd.DataFrame(data))
data={"a":1,"b":2,}
print(pd.Series(data))
daat=[1,2,3]
print(pd.Series(daat,index=[4,5,6]))
data={"a":1,"b":2}
print(pd.DataFrame(data,index=[10,20]))
print(np.random.rand(3,2))
print(np.random.RandomState(44))
print(pd.DataFrame(np.random.rand(3,2),columns=["foo","bar"],index=["a","b","c"]))
ind=pd.Series([1,2,3,4])
print(np.dtype(ind))
data=pd.Series([0.25,0.5,0.75,1.0],index=["a","b","c","d"])
print(data)
print("a" in data)
print(data.keys())
print(list(data.items()))
data["e"]=1.25
print(data)
print(data["a":"c"])
print(data[(data>0.3)&(data<0.8)])
print(data[["a","e"]])
data=pd.Series(["a","b","c"],index=[1,3,5])
print(data)
print(data[1:3])
print(data.loc[1:2])
print(data.iloc[1])
print(data.iloc[1:3])
area=pd.Series({"Yaba":123,"Somolu":456,"Bariga":789})
pop=pd.Series({"Yaba":12344,"Somolu":45623,"Bariga":17890})
data=pd.DataFrame({"area":area,"pop":pop})
print(data)
print(data["area"])
print(data.area is data["area"])
data["density"]=data["pop"]/data["area"]
print(data)
people=["tumi","folake","james"]
import matplotlib.pyplot as plt
plt.scatter(data["area"],people,s=100,c="green")
plt.title("trial")
#plt.show()
print(data.values)
print(data.index)
data_transposed=data.T
print(data_transposed)
print(data.values[0])
print(data["area"])
print(data.iloc[:3,:2])
print(data)
print(data.loc[:"Somolu",:"pop"]) #slicing with the loc (keywordS)
#print(data.ix[:3,:"pop"]) #omo this says no attribute ix for Data Frame
print(data.loc[data["density"]>100,["pop","density"]])
print(data.iloc[0,2])
data.iloc[0,2]=90
rng=np.random.RandomState(42)
ser= pd.Series(rng.randint(0,10,4,))
print(ser)
df=pd.DataFrame(rng.randint(0,10,(3,4)),columns=["A","B","C","D"])
print(df)
print(np.exp(ser))
print(np.exp(df))
print(np.sin(df*np.pi/4))
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,'California': 423967})
population = pd.Series({'California': 38332521, 'Texas': 26448193,'New York': 19651127})
states=pd.DataFrame({"area":area,"population":population})
print(states)
print(population/area)
A= pd.Series([2,4,6],index=[0,1,2])
B=pd.Series([1,3,5],index=[1,2,3])
print(A+B)
print(A.add(B,fill_value=0))
A=pd.DataFrame(rng.randint(0,20,(2,2),),columns=list("AB"))
print(A)
B=pd.DataFrame(rng.randint(0,10,(3,3),),columns=list("BAC"))
print(B)
print(A.add(B))
fill=A.stack().mean()
print(A.add(B,fill_value=fill))
A=rng.randint(10,size=(3,4))
print(A)
print(A-A[0])
df=pd.DataFrame(A,columns=list("QRST"))
print(df)
print(df-df.iloc[0 ])
print(df.sub(df["R"],axis=0))
half_row=print(df.iloc[0,::2])
vals2=np.array([1,np.nan,3,4])
print(vals2)
print(1+np.nan)
print(np.sum(vals2))
print(np.min(vals2))
print(np.max(vals2))
print(np.nansum(vals2),np.nanmin(vals2),np.nanmax(vals2))
print(pd.Series([1,np.nan,2,None]))
print(pd.Series(range(2),index=[1,2], dtype=int))
data=pd.Series([1,np.nan,"Hello",None])
print(data)
print(data.isnull())
print(data[data.notnull()])
print(data.dropna())
#print(data.fillna(3))
data=pd.DataFrame([[1,np.nan,2],[2,3,5],[np.nan,4,6]])
print(data)
print(data.dropna())
print(data.dropna(axis="columns"))
data[3]=np.nan
print(data)
print(data.dropna(axis="columns",how="all"))
print(data.dropna(axis="rows",how="all"))
print(data.dropna(axis="columns",thresh=3))
data=pd.Series([1,np.nan,2, None,3], index=list("abcde"))
print(data)
print(data.fillna(0))
print(data.fillna(method="ffill"))
print(data.fillna(method="bfill"))
df=pd.DataFrame([[1,np.nan,2],[2,3,5],[np.nan,4,6]])
df[3]=np.nan
print(df)
print(df.fillna(method="bfill",axis="columns"))
print(df.fillna(method="ffill",axis="rows"))
index=[("california",2000),("california",2010),("New York",2000),("New York",2010),("Texas",2000),("Texas",2010)]
population=[33234,34424,5532,32244,53233,42422]
pop=pd.Series(population,index=index)
print(pop)
print(pop[[i for i in pop.index if i[1] ==2010]])
print(pop[[i for i in pop.index if i[0] =="california"]])
for i in pop.index:
    if i[1]==2010:
        print(pop[i])
index=pd.MultiIndex.from_tuples(index)
print(index)
pop=pop.reindex(index)
print(pop)
print(pop["Texas",:])
pop_df= pop.unstack()
print(pop_df)
print(pop_df.stack())
print(pop_df.stack().mean())
pop_df=pd.DataFrame({"total":pop,"under18":[3422,5544,6422,6676,2432,8646]})
print(pop_df)
print(pop_df.unstack())
f_u18= pop_df["under18"]/pop_df["total"]
print(f_u18.unstack())
df=pd.DataFrame(np.random.rand(4,2), index=[["a","a","b","b"],[1,2,1,2]],columns=["data1","data2"])
print(df)
data = {('California', 2000): 33871648,('California', 2010): 37253956,('Texas', 2000): 20851820,('Texas', 2010): 25145561,('New York', 2000): 18976457,('New York', 2010): 19378102}
data_series=pd.Series(data)
index=pd.MultiIndex.from_tuples(data)
data_df=data_series.reindex(index)
print(data_df.unstack())
pop.index.names=["states","year"]
print(pop)
print(pop["california",2000])
a=pd.MultiIndex.from_arrays([["a","b","a","b"],[1,2,1,2]])
multiarray_test=pd.DataFrame(pop,index=a)
print(multiarray_test)
#print(multiarray_test.unstack())
print(pd.MultiIndex.from_product([["a","b"],[1,2]])) #hierarchical indices and columns
index=pd.MultiIndex.from_product([[2013,2014],[1,2]],names=["year","visit"])
columns=pd.MultiIndex.from_product([["Bob","Guido","Sue"],["HR","Temp"]],names=["subject","type"])
data=np.round(np.random.randn(4,6),1) #mock data
print(data)
print(data[:,::2])
data[:,::2]*=10
data+=37
health_data=pd.DataFrame(data,index=index,columns=columns)
print(health_data)
print(health_data.iloc[:2,:4])
print(health_data.loc[:,"Guido"])
print(health_data.loc[:,("Guido","HR")])
idx=pd.IndexSlice
print(health_data.loc[idx[:,1],idx[:,"HR"]])
index=pd.MultiIndex.from_product([["a","c","b"],[1,2]],names=["Char","int"])
data= pd.Series(np.round(np.random.rand(6),1),index=index)
print(data)
data=data.sort_index()
print(data)
print(data["a":"b"])
print(pop.unstack(level=0))
print(pop.unstack(level=1))
print("aa")
pop_flat=pop.reset_index(name="population")
print(pop_flat)
print(pop_flat.set_index(["states","year"]))
print(health_data)
data_mean=health_data.mean(level="year")
print(data_mean)
print(health_data.mean(axis="columns",level="type"))
data_mean2=data_mean.mean(axis="columns",level="type")
print(data_mean2)

def make_df(cols,ind):
    data= {c:[str(c)+ str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)
print(make_df("ABC",range(3)))
ser1=pd.Series(["A","B","C"],index=[1,2,3])
ser2=pd.Series(["D","E","F"],index=[4,5,6])
print(pd.concat([ser1,ser2]))
x=make_df("AB",[0,1])
y=make_df("AB",[2,3])
y.index=x.index
print(y.index)
print(pd.concat([x,y]))
try:
    pd.concat([x,y], verify_integrity=True)
except ValueError as e:
    print("ValueError:",e)
print(pd.concat([x,y],ignore_index=True))
print(pd.concat([x,y],keys=["x","y"]))
df5=make_df("ABC",[1,2])
df6=make_df("BCD",[3,4])
print(df5)
print(df6)
print(pd.concat([df5,df6]))
print(pd.concat([df5,df6],join="inner"))
#print(pd.concat([df5,df6],join_axes=[df5.columns]))
print(df5.append(df6))
df1=pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2= pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df3=pd.merge(df2,df1)
print(df3)
print(pd.concat([df1,df2],axis="columns"))
df4=pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],'supervisor': ['Carly', 'Guido', 'Steve']})
print(pd.merge(df3,df4))
print(pd.merge(df1,df2,on="employee"))
print(pd.merge(df1,df2))
df3=pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],'salary': [70000, 80000, 120000, 90000]})
print(pd.merge(df1,df3,left_on="employee",right_on="name"))
print("aaaaaa")
print(pd.merge(df1,df3,left_on="employee",right_on="name").drop("name",axis="columns"))
print(df1)
df1a=df1.set_index("employee")
df2a=df2.set_index("employee")
print(df1a); print(df2a)
df1aa=df1a.reset_index("employee")
print(df1aa)
print(pd.merge(df1a,df2a,left_index=True,right_index=True))
print(df1a.join(df2a))
print(pd.merge(df1a,df3,left_index=True,right_on="name"))
df6= pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],'food': ['fish', 'beans', 'bread']},columns=["name","food"])
df7= pd.DataFrame({'name': ['Mary', 'Joseph'],'drink': ['wine', 'beer']},columns=['name', 'drink'])
print(pd.merge(df6,df7,how="inner"))
print(pd.merge(df6,df7,how="outer"))
print(pd.merge(df6,df7,how="left"))
print(pd.merge(df6,df7,how="right"))
df8=pd.DataFrame({"name":["Bob","Jake","Lisa","Sue"],"rank":[1,2,3,4]})
df9=pd.DataFrame({"name":["Bob","Jake","Lisa","Sue"],"rank":[3,1,4,2]})
print(pd.merge(df8,df9,on="name"))
print(pd.merge(df8,df9,on="name",suffixes=["_L","_R"]))
pop=pd.read_csv('data-USstates-master\state-population.csv')
areas=pd.read_csv('data-USstates-master\state-areas.csv')
abbrev=pd.read_csv('data-USstates-master\state-abbrevs.csv')
neww=pd.DataFrame({"statee":["a","b","c","d"],"abbreviation":["QW","QE","QR","QT"]})
print(pop.head())
print(areas.head())
print(abbrev.head())
merged=pd.merge(pop,abbrev,how="outer",left_on="state/region",right_on="abbreviation")
merged=merged.drop("abbreviation",axis="columns")
print(merged.head())
print(merged.isnull().any())
print(merged[merged["population"].isnull()].head())
print(merged[merged["state"].isnull()].head())
merged.loc[merged["state/region"]=="PR","state"]="Puerto Rico"
merged.loc[merged["state/region"]=="USA","state"]="United states"
print(merged.isnull().any())
final=pd.merge(merged,areas,on="state",how="left")
print(final.head())
print(final.isnull().any())
final[final["population"].isnull()]= 200
print(final["population"].isnull().any())
final_population= final["population"]
print(final_population.loc[2449:2491])
print(final.isnull().any())
print(123344)
print(final["state"][final["area (sq. mi)"].isnull()].unique())
final.dropna(inplace=True)
print(final.isnull().any())
print(final.head())
data2010= final.query("year==2010")
print(data2010)
data2010_totalages=final.query("year==2010 & ages=='total'")
print(data2010_totalages.head())
print("aa")
print(data2010.set_index("state",inplace=True))
density=data2010["population"]/data2010["area (sq. mi)"]
density.sort_values(ascending=False,inplace=True)
print(density.head())
print(density.tail())
import seaborn as sns
planets= sns.load_dataset('planets')
print(planets)
print(planets.head())
rng=np.random.RandomState(42)
ser=pd.Series(rng.rand(5))
print(ser)
print(np.sum(ser)); print(ser.sum())
print(np.mean(ser)); print(ser.mean())
df=pd.DataFrame({"A":rng.rand(5),"B":rng.rand(5)},index=[11,12,13,14,15])
print(df)
print(df.mean())
print(df.mean(axis="columns"))
print(planets[planets["mass"].isnull()])
print("aa")
print(planets.isnull().any().count())
print(planets.describe())
print(planets.dropna().describe())
input=pd.DataFrame({"A":[1,4],"B":[2,5],"C":[3,6]})
print(input)
input_A=input["A"]
input_B=input["B"]
input_C=input["C"]
inputA=input_A.sum()
inputB=input_B.sum()
inputC=input_C.sum()
inputdA=pd.DataFrame({"A":inputA},index=[0])
inputdB=pd.DataFrame({"B":inputB},index=[0])
inputdC=pd.DataFrame({"C":inputC},index=[0])
mergedAB=pd.merge(inputdA,inputdB,left_on="A",right_on="B",left_index=True,right_index=True)
print(mergedAB.join(inputdC))
#another way without the use of merge
#print(inputdA)
#input_A=pd.Series(input_A.sum())
#input_B=pd.Series(input_B.sum())
#input_C=pd.Series(input_C.sum())
#print(pd.DataFrame({"A":input_A,"B":input_B,"C":input_C}))
#print(pd.merge(input_A,input_B,left_on="A",right_on="B"))
df=pd.DataFrame({"key":["A","B","C","A","B","C"],"data":range(6)},columns=["key","data"])
print(df)
print(df.groupby("key"))
print(df.groupby("key").sum())
print(planets)
print(planets.groupby("method"))
print(planets.groupby("method")["orbital_period"])
print(planets.groupby("method")["orbital_period"].median())
#planets= planets.set_index("method")
#print(data2010.set_index("state",inplace=True))
#planets_method=planets["orbital_period"]
#print(planets_method.median())
print(planets.groupby("method")["year"].describe())
rng=np.random.RandomState(0)
df=pd.DataFrame({"key":["A","B","C","A","B","C"],"data1":range(6),"data2":rng.randint(0,10,6)})
print(df)
print(df.groupby("key").aggregate([min,np.median,max,np.std]))
def filter_func(x):
    return x['data2'].std()>4
print(df); print(df.groupby("key").std())
print(df.groupby('key').filter(filter_func))
def filter_fun(y):
    return y["data1"].mean()>=2
print(df); print(df.groupby('key').mean())
print(df.groupby('key').filter(filter_fun))
print(df.groupby('key').transform(lambda x: x - x.mean()))
print(lambda x:x+2)

def norm_by_data2(x):
    x['data1']/=x['data2'].sum()
    return x
print(df);print(df.groupby('key').apply(norm_by_data2))
l=[0,1,0,1,2,0]
print(df);print(df.groupby(l)); print(df.groupby(l).sum())
df2= df.set_index('key')
mapping={'A':"Vowel","B":"Word","C":'consonant'}
print(df2.groupby(mapping).sum())
mapping={'A':"Vowel","B":"consonant","C":'consonant'}
print(df2.groupby(mapping).sum())
print(df2.groupby(str.lower).mean())
print(df2.groupby([str.lower,mapping]).mean())
print(df2.groupby(['key',mapping]).mean())
print(planets['year'])
decade= 10 * (planets['year']//10)
print(decade)
decade= decade.astype(str) + 's'
print(decade)
decade.name='decade'
print(planets['method'])
print(planets)
print(planets.groupby(['method',decade])['number'].sum().unstack().fillna(0))
import seaborn as sns
titanic= sns.load_dataset('titanic')
print(titanic.head())
