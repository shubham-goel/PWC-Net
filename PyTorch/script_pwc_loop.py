import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
from multiprocessing.connection import Listener, Client
import pickle

import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())


def compute_flow(net, img0, img1):
	# img0: B,h,w,BGR
	# img1: B,h,w,BGR
	assert(len(img0.shape)==4)

	# rescale the image size to be multiples of 64
	divisor = 64.
	B = img0.shape[0]
	H = img0.shape[1]
	W = img0.shape[2]

	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)

	im_all = np.concatenate((img0, img1), axis=0)
	im_all_new = np.zeros((2*B,H_,W_,3), dtype=im_all.dtype)
	for i in range(im_all.shape[0]):
		im_all_new[i] = cv2.resize(im_all[i], (W_, H_))
	im_all = im_all_new

	# for _i, _inputs in enumerate(im_all):

	# 	im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
	# 	im_all[_i] = torch.from_numpy(im_all[_i])
	# 	im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
	# 	im_all[_i] = im_all[_i].float()

	if im_all.dtype == np.uint8:
		im_all = 1.0 * im_all/255.0

	im_all = np.transpose(im_all, (0,3,1,2))
	im_all = torch.from_numpy(im_all).float()
	img0 = im_all[:B,:,:,:]
	img1 = im_all[B:,:,:,:]
	im_all = torch.autograd.Variable(torch.cat((img0, img1),1).cuda(), volatile=True)

	flo = net(im_all)
	flo = flo * 20.0
	flo = flo.cpu().data.numpy() # B,2,h,w

	# scale the flow back to the input size
	flo = np.transpose(flo, (0,2,3,1))
	u_ = flo[:,:,:,0]
	v_ = flo[:,:,:,1]
	u_new = np.zeros((B,H,W), dtype=u_.dtype)
	v_new = np.zeros((B,H,W), dtype=v_.dtype)
	for i in range(B):
		u_new[i] = cv2.resize(u_[i],(W,H))
		v_new[i] = cv2.resize(v_[i],(W,H))
	u_ = u_new
	v_ = v_new
	u_ *= W/ float(W_)
	v_ *= H/ float(H_)
	flo = np.stack((u_,v_), axis=3)
	assert(flo.shape == (B,H,W,2))
	return flo

def server_loop(net):
	if len(sys.argv) > 1:
		port = int(sys.argv[1])
	else:
		port = 6000
	address = ('localhost', port)     # family is deduced to be 'AF_INET'
	listener = Listener(address, authkey=b'secret password')
	while True:
		print('Waiting for new connection...')
		conn = listener.accept()
		print('Connection accepted from', listener.last_accepted)
		try:
			while True:
				msg = conn.recv_bytes()  # B,h,w,ch
				(img0, img1) = pickle.loads(msg)

				print('New Message:', img1.shape, img1.dtype)

				# Compute flow from images
				flow = compute_flow(net, img0, img1)

				print('Responding:', flow.shape, flow.dtype)
				conn.send(flow)
		except EOFError:
			print('EOFError!!')

if __name__ == '__main__':
	pwc_model_fn = './pwc_net.pth.tar'
	net = models.pwc_dc_net(pwc_model_fn)
	net = net.cuda()
	net.eval()

	server_loop(net)
