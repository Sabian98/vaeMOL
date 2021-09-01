
import numpy as np

import os


f=open("/scratch/trahman2/list.txt")
fr=f.readlines()
f.close()


for elem in fr[0:10]:
	

	cont=elem.strip("\n")
	arr=np.loadtxt("/scratch/trahman2/maps/maps_original/"+cont)
	# print(arr.size)
	# if arr.size>=128:
	# 	map_arr=arr[:128]
	# 	np.savetxt("/scratch/trahman2/maps/res_maps_128/"+cont,map_arr)

	print(np.shape(arr))

	# # break
	
	'''if (len(arr)<=600):
	# 	arr=arr[:128]
		map_arr=[]
		for i in range (len(arr)):
			res_arr=[]
			for j in range(len(arr)):
				res_arr.append(np.linalg.norm(arr[i]-arr[j])>8)
			map_arr.append(res_arr)
		np.savetxt("/scratch/trahman2/maps/maps_original/"+cont,map_arr)'''
		# print(len(map_arr[0]))
		# print(len(map_arr))
# b=np.zeros(600,600)


# for elem in fr:
	
# 	cont=elem.strip("\n")
	
# 	arr=np.loadtxt("/scratch/trahman2/residues/"+cont)
	
# 	result=np.ones((600,), dtype=int)
# 	result=result*20
# 	result[:arr.size] = arr
# 	np.savetxt("/scratch/trahman2/maps/padded_res_maps/"+cont,result)
	# print(result)

	







	


	
	

