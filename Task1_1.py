import sys
import random
import numpy 


print(sys.platform)
k = input("Input K")
N = input("Input N")

i = 0
list = []
s, n = set(), 0
while i < int(k):
    list.append(random.randrange(1,int(N)))
    i=i+1
print(list)

for elem in list:
    if elem not in s:
        s.add(elem)
        n+= 1

print(n)

list2 = numpy.zeros(int(k)) 
for elem in list: 
    list2[elem-1] = 1
count = 0 
for elem in list2: 
    count+=elem
    
print(count)
