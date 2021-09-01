import torch
from models_vae import Generator, Discriminator, EncoderVAE
import numpy as np
from scipy.stats.stats import pearsonr,wasserstein_distance
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import math
import pickle

# from molecules_code import classes_16
# from classes_16 import Generator




class measures:
	def __init__(self,data_path,enc_path,length,epoch,tp,model_type):
		self.contact=0
		self.g_conv_dim=[128, 256, 512]
		self.z_dim = 8
		self.ngpu=2
		self.epoch=epoch
		self.type=tp
		self.model_type=model_type
		self.vertexes=length
		self.bond_num_types=2
		self.atom_num_types=21
		self.dropout_rate=0.
		self.batch_size=64
		self.enc_path=enc_path
		self.data_path=data_path
		self.post_method="usual"
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.load_model()

		#for molecules
		self.nz=100
		self.fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)

	def load_model(self):
		self.data = np.load(self.data_path,allow_pickle='TRUE').item()
	# print(len(data.values()))
		# self.num_steps=1
		self.num_steps=len(self.data.values())//self.batch_size
		if self.model_type=="graphVAE":

			self.vae_stats()





	def Bhattacharyya(self,x,y):

		plt.figure(figsize=(8,4), dpi=80) 
		cnt_x = plt.hist(x, bins=20,width=0.5)
		cnt_y = plt.hist(y, bins=20)
		x_=cnt_x[0]/len(x)   # No. of points in bin divided by total no. of samples.
		y_=cnt_y[0]/len(y)    
		BC=np.sum(np.sqrt(x_*y_))
		plt.close()
		return -np.log(BC)



	def stats(self,original_distances,generated_distances):
		print("stats for epoch "+ str(self.epoch)+" is\n")
		# print("PCC is \n")
		# print(str(pearsonr(original_distances,generated_distances))+"\n")
		print("EMD is \n")
		print(str(wasserstein_distance(original_distances,generated_distances))+"\n")
		print("BD is \n")
		print(str(self.Bhattacharyya(original_distances,generated_distances))+"\n")

	def short_range(self,seq_length,edges,th):
		if seq_length==0:
			seq_length=200
		count=0

		for index in range(seq_length):
			arr2=edges[index][index+2:index+5]
				
			count+=arr2[np.where(arr2 == th)].size
			
		score=count/seq_length
		if  math.isnan(score)==False:
			ret= count/seq_length
		else:
			ret=0
		return ret

	def long_range(self,seq_length,edges,th):
		count=0
		if seq_length==0:
			seq_length=200

		for index in range(seq_length):
			arr2=edges[index][index+2:]
				
			count+=arr2[np.where(arr2 == th)].size
			
		score=count/seq_length
		if  math.isnan(score)==False:
			ret= count/seq_length
		else:
			ret=0
		return ret
		


	def load_original(self):

		original_arr=[]
		for item in self.data.values():
			if self.type=="long":

				original_arr.append(self.long_range(np.argmax(item[1]>19),item[0],0))
			elif self.type=="short":
				original_arr.append(self.short_range(np.argmax(item[1]>19),item[0],0))
		return original_arr

	def vae_stats(self):

		# print(self.num_steps)
		self.decoder = Generator(self.g_conv_dim, self.z_dim, self.vertexes, self.bond_num_types,
                                 self.atom_num_types, self.dropout_rate).to(self.device)

		self.decoder.load_state_dict(torch.load(self.enc_path, map_location=lambda storage, loc: storage))


		
		generated_distances=[]
		backbone=[]
		
		
		for _ in range(self.num_steps):
			z = self.sample_z(self.batch_size)
			z = torch.from_numpy(z).to(self.device).float()
			edges_logits, nodes_logits = self.decoder(z)
			edges, nodes = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
			torch.save(edges.cpu(),"/scratch/trahman2/edges.pt")
			torch.save(nodes.cpu(),"/scratch/trahman2/nodes.pt")
			break
			'''
			edges=edges.cpu().numpy()
			nodes=nodes.cpu().numpy()

			
			for elem in zip(edges,nodes):
				# print(np.argmax(elem[1]>19))
				if self.type=="long":
					generated_distances.append(self.long_range(np.argmax(elem[1]>19),elem[0],0))
				elif self.type=="short":
					generated_distances.append(self.short_range(np.argmax(elem[1]>19),elem[0],0))
				else:
					score=self.backbone(elem[0],0,np.argmax(elem[1]>19))
					if math.isnan(score)==False :
						backbone.append(score)
		# print(np.mean(backbone))
		# print(len(backbone))
			
		if self.type!="backbone":
			original_distances=self.load_original()
			if (len(generated_distances)>len(original_distances)):
				generated_distances=generated_distances[:len(original_distances)]
			else:
				original_distances=original_distances[:len(generated_distances)]
		# original_distances=list(filter(lambda num: num != 0, original_distances))
		# generated_distances=list(filter(lambda num: num != 0, generated_distances))
		# np.savetxt("/scratch/trahman2/original_distances.txt",original_distances)
		# np.savetxt("/scratch/trahman2/dist.txt",generated_distances)
		self.histo(original_distances,generated_distances)'''


	@staticmethod
	def postprocess_logits(inputs, method, temperature=1.):
		def listify(x):
 			return x if type(x) == list or type(x) == tuple else [x]

		def delistify(x):
			return x if len(x) > 1 else x[0]

		if method == 'soft_gumbel':
			softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
		elif method == 'hard_gumbel':
			softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
		else:
			softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

		return [delistify(e) for e in (softmax)]


	def backbone(self,elem,th,elem_len):

		count=0
		for index in range(elem_len-1):

			if elem[index][index+1]==th:
				count+=1

		
		return count/(elem_len-1)



	def get_gen_mols(self, n_hat, e_hat, method):
		(edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
		edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]

		return edges_hard, nodes_hard

	def sample_z(self, batch_size):
			return np.random.normal(0, 1, size=(batch_size, self.z_dim))

	def histo(self,real,fake):
		# bins = np.linspace(0, 200,10)
		# plt.hist(fake, bins, alpha=0.5, label='Generated',color='r')
		# plt.hist(real, bins, alpha=0.5, label='Input',color='b')
		num_bins=100
		n, bins, patches = plt.hist(fake, num_bins, facecolor='red', alpha=0.5, label='Generated')
		n2, bins2, patches2 = plt.hist(real, num_bins, facecolor='blue', alpha=0.5, label='Input')
		plt.legend()
		plt.xlim([-0.75, 6.75])
		plt.savefig("/scratch/trahman2/histo_200_"+str(self.epoch)+"_"+str(self.type)+".png"
			,dpi=300,bbox_inches="tight",pad_inches = 0)
		plt.close()



	# def get_sample(self):

len_arr=[200]
epochs=[50]
models=["graphVAE"]
ms_type="long"

for model_type in models:

	for length in len_arr:

		for epoch in epochs:

			ms=measures("/scratch/trahman2/my_file_200.npy","/scratch/trahman2/saved_models/models_"+
				str(length)+"/"+str(epoch)+"-decoder.ckpt",length,epoch,ms_type,model_type)












