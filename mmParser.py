
import numpy as np
import os
from Bio.PDB.PDBParser import PDBParser
p = PDBParser(PERMISSIVE=1)




# line="ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL"
# fr=line.split(",")
# atom_dict={}
# val=0
# for elem in fr:
#     atom_dict[elem]=val
#     val+=1

file=open("/scratch/trahman2/list.txt")

f=file.readlines()
file.close()
'''

data_dic={}
for item in f[:10]:
    tup=[]
    cont=item.strip("\n")
    structure_id = cont[:4]
    adj=np.loadtxt("/scratch/trahman2/maps/maps_original/"+structure_id+".txt")
    nodes=np.loadtxt("/scratch/trahman2/maps/padded_res_maps/"+structure_id+".txt")

    adj=adj[:200,:200]
    nodes=
    
    # print(np.shape(nodes))
    print(np.argmax(nodes>19))
    # tup.append(nodes[:200])

    data_dic[structure_id]=tup'''

    

# np.save('/scratch/trahman2/my_file_200.npy', data_dic)
dict2={}
dict1=np.load('/scratch/trahman2/my_file_200.npy',allow_pickle='TRUE').item()
for key,val in dict1.items():


    ind=np.argmax(val[1]>19)
    if ind==0:
        ind=200
    tup=[]
    tup.append(val[0][:ind,:ind])
    dict2[key]=tup
    print(np.shape(tup[0]))
    


np.save('/scratch/trahman2/variable_length.npy', dict2)

'''
for elem in fr[len(fr)//2:]:

    
    arr=[]
    cont=elem.strip("\n")
    structure_id = cont[:4]
    filename = "/scratch/trahman2/ext_structures/"+cont#folder containing unzipped pdb files
    structure = p.get_structure(structure_id, filename)#get structure
    model = structure[0]#get the first model only
    for chain in model:
        if chain.get_id()=="A":#search for a particular chain ID
            for residue in chain:
                if residue.get_resname() in atom_dict:#get residue name
                    for atom in residue:
                        if atom.get_id()=="CA":#get only C- alpha atom
                            arr.append(atom.get_coord().tolist())#to get the coordinates of the aforemnetioned atom
                            # arr.append(atom_dict[residue.get_resname()])#to get the residue name of the aforemnetioned atom
                            # break





    # np.savetxt("/scratch/trahman2/residues/"+structure_id+".txt",arr)#save 
    np.savetxt("/scratch/trahman2/vectors/"+structure_id+".txt",arr)

'''








