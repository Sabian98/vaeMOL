from data import dataset
import numpy as np
import torch
import os
import time


import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models_vae import Generator, Discriminator, EncoderVAE

class solver:

	def __init__(self,path,num_vertices,dec_path,epoch_arr):
		self.mode='train'
		self.post_method="usual"
		self.num_epochs=1
		self.z_dim = 8
		self.dec_path=dec_path
		self.bond_num_types=2
		self.atom_num_types=21
		self.m_dim = self.atom_num_types
		self.b_dim = self.bond_num_types
		self.f_dim = 0
		self.dropout=0
		self.g_conv_dim=[128, 256, 512]
		self.d_conv_dim=[[128, 64], 128, [128, 64]]
		self.vertexes=num_vertices
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.data = np.load(path,allow_pickle='TRUE').item()
		self.batch_size=64
		self.num_steps = (len(self.data) // self.batch_size)
		self.ds=dataset()
		self.dropout_rate=0
		self.losses_dict={}
		self.losses_dict['recon_loss']=[]
		self.losses_dict['kl_loss']=[]
		self.epoch_arr=epoch_arr
		self.kl_la = 1.
		self.g_lr=0.01
		self.d_lr=0.001
		self.resume_epoch=0
		self.build_model()

	def build_model(self):
		self.encoder = EncoderVAE(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.z_dim,
                                  with_features=False, f_dim=self.f_dim, dropout_rate=self.dropout_rate).to(self.device)
		self.decoder = Generator(self.g_conv_dim, self.z_dim, self.vertexes, self.bond_num_types,
                                 self.atom_num_types, self.dropout_rate).to(self.device)
		self.vae_optimizer = torch.optim.RMSprop(list(self.encoder.parameters()) +
                                                 list(self.decoder.parameters()), self.g_lr)

	def restore_model(self,epoch):

		e_path = os.path.join(self.dec_path, '{}-encoder.ckpt'.format(epoch))
		d_path = os.path.join(self.dec_path, '{}-decoder.ckpt'.format(epoch))

		self.encoder.load_state_dict(torch.load(e_path, map_location=lambda storage, loc: storage))
		self.decoder.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))

	def update_lr(self, g_lr, d_lr):
		"""Decay learning rates of the generator and discriminator."""
		for param_group in self.g_optimizer.param_groups:
			param_group['lr'] = g_lr
		for param_group in self.d_optimizer.param_groups:
			param_group['lr'] = d_lr

	def reset_grad(self):
		"""Reset the gradient buffers."""
		self.vae_optimizer.zero_grad()
		# self.v_optimizer.zero_grad()

	def label2onehot(self, labels, dim):
		"""Convert label indices to one-hot vectors."""
		out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
		out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
		return out

	def sample_z(self, batch_size):
		return np.random.normal(0, 1, size=(batch_size, self.z_dim))

	def get_reconstruction_loss(self, n_hat, n, e_hat, e):
		# This loss cares about the imbalance between nodes and edges.
		# However, in practice, they don't work well.
		# n_loss = torch.nn.CrossEntropyLoss(reduction='none')(n_hat.view(-1, self.m_dim), n.view(-1))
		# n_loss_ = n_loss.view(n.shape)
		# e_loss = torch.nn.CrossEntropyLoss(reduction='none')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
		# e_loss_ = e_loss.view(e.shape)
		# loss_ = e_loss_ + n_loss_.unsqueeze(-1)
		# reconstruction_loss = torch.mean(loss_)
		# return reconstruction_loss

		n_loss = torch.nn.CrossEntropyLoss(reduction='mean')(n_hat.view(-1, self.m_dim), n.view(-1))
		e_loss = torch.nn.CrossEntropyLoss(reduction='mean')(e_hat.reshape((-1, self.b_dim)), e.view(-1))
		reconstruction_loss = n_loss + e_loss
		return reconstruction_loss

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


	def get_gen_mols(self, n_hat, e_hat, method):
		(edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
		edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
		# mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
  #               for e_, n_ in zip(edges_hard, nodes_hard)]
		return edges_hard, nodes_hard

	def backbone_score(self,arr,th):
		scores=[]
		for elem in arr:
			count=0
			for i in range(len(elem)-1):
				for j in range(len(elem)-1):
					if i==j:
						count+=(elem[i][j+1]==th)
			scores.append(count)
		print(np.mean(scores))




	@staticmethod
	def get_kl_loss(mu, logvar):
		kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
		return kld_loss

	def train(self):
		start_epoch=0
		arr=[]
		for item in self.data.values():
			arr.append(item)
		self.ds.generate_train_validation(0.0001,arr)

		if self.resume_epoch:
			start_epoch=self.resume_epoch
			self.restore_model(self.resume_epoch)
		t1=time.time()


		for i in range(start_epoch, self.num_epochs):
				self.train_or_valid(epoch_i=i, train_val_test='train')
				# self.train_or_valid(epoch_i=i, train_val_test='val')
				# self.train_or_valid(epoch_i=i, train_val_test='sample')
		t2=time.time()
		print(t2-t1)


		'''
		print('Time needed for '+ str(self.num_epochs)+ ' epoch(s) is '+ str(t2-t1))
		plt.plot(self.losses_dict["recon_loss"],label="recon_loss")
		plt.plot(self.losses_dict["kl_loss"],label="kl_loss")
		plt.legend()

		plt.savefig("/scratch/trahman2/loss_figs/vae_loss_fig_"+str(self.num_epochs)+"_"+str(self.vertexes)+".png",bbox_inches="tight",pad_inches = 0,dpi=300)
		'''
		
	def save_checkpoints(self, epoch_i):
		dec_path = os.path.join(self.dec_path, '{}-decoder.ckpt'.format(epoch_i + 1))
		enc_path=os.path.join(self.dec_path, '{}-encoder.ckpt'.format(epoch_i + 1))
		torch.save(self.decoder.state_dict(), dec_path)
		torch.save(self.encoder.state_dict(), enc_path)


	def train_or_valid(self, epoch_i, train_val_test='val'):
		the_step = self.num_steps
		if train_val_test == 'val':
			if self.mode == 'train':
				the_step = 1
			print('[Validating]')

		if train_val_test == 'sample':
			if self.mode == 'train':
				the_step = 1
			print('[Sampling]')

		for a_step in range(the_step):####
			if(train_val_test=="val"):
				data=self.ds.next_valid_batch()
				# print(len(data))
			elif (train_val_test=="train"):
				data=self.ds.next_train_batch(self.batch_size)
			elif (train_val_test=="sample"):
				z = self.sample_z(self.batch_size)
				z = torch.from_numpy(z).to(self.device).float()


			nodes=[]
			edges=[]

			for elem in data:####


				nodes.append(elem[1])

				edges.append(elem[0])

				

				
			
			if train_val_test == 'train' or train_val_test == 'val':
				a = torch.tensor(edges).to(self.device).long()  # Adjacency.
				x = torch.tensor(nodes).to(self.device).long()  # Nodes.
				a_tensor = self.label2onehot(a, self.b_dim)
				x_tensor = self.label2onehot(x, self.m_dim)

			
			# print("tensor shapes before being fed ")

			
			if train_val_test == 'train' or train_val_test == 'val':
				z, z_mu, z_logvar = self.encoder(a_tensor, None, x_tensor)
			
			
			
			edges_logits, nodes_logits = self.decoder(z)

			if train_val_test == 'train' or train_val_test == 'val':
				recon_loss = self.get_reconstruction_loss(nodes_logits, x, edges_logits, a)
				kl_loss = self.get_kl_loss(z_mu, z_logvar)
				self.losses_dict['kl_loss'].append(kl_loss.item())
				self.losses_dict['recon_loss'].append(recon_loss.item())
				loss_vae = recon_loss + self.kl_la * kl_loss
                # f = torch.from_numpy(f).to(self.device).float()
				vae_loss_train =  loss_vae

				if train_val_test == 'train':
					self.reset_grad()
					vae_loss_train.backward()
					# loss_v.backward()
					self.vae_optimizer.step()

		if epoch_i+1 in self.epoch_arr:
			# z = self.sample_z(self.batch_size)
			# z = torch.from_numpy(z).to(self.device).float()
			# edges_logits, nodes_logits = self.decoder(z)
			# edges, nodes = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
			self.save_checkpoints(epoch_i)




					
sol=solver('/scratch/trahman2/my_file_200.npy',200,"/scratch/trahman2/saved_models/models_200/",[10])

sol.train()

# print(t2-t1)












