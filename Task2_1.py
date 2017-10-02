
import numpy

def MaxSequence(list_first,list_second):

  n = len(list_first)
  k = len(list_second)
  if n == 0 or k == 0:
    return 0


  A = numpy.zeros((n+1,k+1))


  for i in range(0,n):
      for j in range(0,k):
        if list_first[i] == list_second[j]:
          A[i+1,j+1] = 1 + A[i,j]
        else:
          A[i + 1, j + 1] = numpy.maximum(A[i,j+1],A[i+1,j])
  print(A)
  if A[n,k] == 0:
    return 1
  return A[n,k]


print(MaxSequence([3, 5,12,7],[18, 5, 3, 5, 12,7]))

