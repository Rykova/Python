import sys
import random


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
        n += 1

print(n)


print(sum)