# -*- coding: UTF-8 -*-

import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt




folder_list=["I","II"]
random_border=10
expand_ratio=0.25

train_list_name="train_list.txt"
test_list_name="test_list.txt"

train_list="train.txt"
test_list="test.txt"

train_boarder=112

def load_images():
	images = []
	for folder_name in folder_list:
		folder=os.path.join("data",folder_name)
		label=os.path.join(folder,"label.txt")
		# print(label)
		with open(label, "r") as f:
			lines = f.readlines()
			for line in lines:
				line=line.strip().split()
				full_direct=os.path.join(folder,line[0])
				if os.path.isfile(full_direct):
					line[0]=folder+"\\"+line[0]
					images.append(line)
					# print(images)
	return images

def crop_image(data_images):
####因有图片中包含多张人脸，所以对图片的人脸进行提取并重新命名。
	new_name=[]
	for i in range(0,len(data_images)):
		face_img=cv2.imread(str(data_images[i][0]),1)
		img_height, img_width, _ = face_img.shape

		orig_x1 = float(data_images[i][1])
		orig_y1 = float(data_images[i][2])
		orig_x2 = float(data_images[i][3])
		orig_y2 = float(data_images[i][4])
		# print(f"face_img:{data_images[i][0]}, orig_x1:{orig_x1},orig_y1:{orig_y1},orig_x2:{orig_x2},orig_y2:{orig_y2}")

		roi_x1, roi_y1, roi_x2, roi_y2 = expend_fc(orig_x1, orig_y1, orig_x2, orig_y2, img_width, img_height,
												   ratio = 0.25)

		face_crop_img=face_img[roi_y1:roi_y2,roi_x1:roi_x2]
		# plt.imshow(face_crop_img)
		new_name.append(str(i)+"-"+str(data_images[i][0][-10::]))
		# print(data_images[i][0])
		cv2.imwrite(f"./data/face/{new_name[i]}",face_crop_img)

	return new_name

def save_train_txt(data_images,new_name):
	train_len = int((float(len(data_images))) * 0.9)
	for i in range(0, train_len):
		face_img = cv2.imread(str(data_images[i][0]), 1)
		img_height, img_width, _ = face_img.shape

		orig_x1 = float(data_images[i][1])
		orig_y1 = float(data_images[i][2])
		orig_x2 = float(data_images[i][3])
		orig_y2 = float(data_images[i][4])
		print(f"face_img:{data_images[i][0]}, orig_x1:{orig_x1},orig_y1:{orig_y1},orig_x2:{orig_x2},orig_y2:{orig_y2}")
		roi_x1, roi_y1, roi_x2, roi_y2 = expend_fc(orig_x1, orig_y1, orig_x2, orig_y2, img_width, img_height,
												   ratio = 0.25)
		images_all = []

		images_all.append(roi_x1)
		images_all.append(roi_y1)
		images_all.append(roi_x2)
		images_all.append(roi_y2)

		line=new_name[i]

		x = map(float, data_images[i][5::2])
		y = map(float, data_images[i][6::2])
		landmarks = list(zip(x, y))
		landmarks -= np.array([roi_x1, roi_y1])

		for i in range(0, landmarks.shape[0]):
			for j in range(0, landmarks.shape[1]):
				images_all.append(landmarks[i, j])

		write_all = (str(images_all)[1:-1]).replace(",", " ")

		write_a = line + " " + write_all
		print(write_a)

		with open("./data/face/train.txt", "a") as f:
			f.write(write_a + "\n")


def save_test_txt(data_images,new_name):
	print(new_name)
	train_len = int((float(len(data_images))) * 0.9)
	test_len=len(data_images)
	print(len(data_images),train_len,test_len)
	for i in range(train_len,test_len):

		face_img = cv2.imread(str(data_images[i][0]), 1)
		img_height, img_width, _ = face_img.shape

		orig_x1 = float(data_images[i][1])
		orig_y1 = float(data_images[i][2])
		orig_x2 = float(data_images[i][3])
		orig_y2 = float(data_images[i][4])
		print(f"face_img:{data_images[i][0]}, orig_x1:{orig_x1},orig_y1:{orig_y1},orig_x2:{orig_x2},orig_y2:{orig_y2}")
		roi_x1, roi_y1, roi_x2, roi_y2 = expend_fc(orig_x1, orig_y1, orig_x2, orig_y2, img_width, img_height,ratio = 0.25)

		images_all = []

		images_all.append(roi_x1)
		images_all.append(roi_y1)
		images_all.append(roi_x2)
		images_all.append(roi_y2)

		line=new_name[i]

		x = map(float,data_images[i][5::2])
		y = map(float,data_images[i][6::2])
		print(x,y)
		landmarks = list(zip(x, y))
		landmarks -= np.array([roi_x1, roi_y1])

		for i in range(0, landmarks.shape[0]):
			for j in range(0, landmarks.shape[1]):
				images_all.append(landmarks[i, j])

		write_all = (str(images_all)[1:-1]).replace(",", " ")

		write_a = line + " " + write_all
		print(write_a)

		with open("./data/face/test.txt", "a") as f:
			f.write(write_a + "\n")

def expend_fc(x1,y1,x2,y2,img_width,img_height,ratio):
	width=x2-x1+1
	height=y2-y1+1
	padding_width=int(width*ratio)
	padding_height=int(height*ratio)
	roi_x1=x1-padding_width
	roi_y1=y1-padding_height
	roi_x2 = x2 + padding_width
	roi_y2 = y2 + padding_height
	roi_x1 = 0 if roi_x1 < 0 else roi_x1
	roi_y1 = 0 if roi_y1 < 0 else roi_y1
	roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
	roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
	return map(int,(roi_x1, roi_y1, roi_x2, roi_y2))

def check_data(data_type,start_img,end_img):
	num=0
	folder = os.path.join("data", "face")
	label = os.path.join(folder, data_type)
	# print(label)
	with open(label, "r") as f:
		lines = f.readlines()
		for line in lines[start_img:end_img]:
			line= line.strip().split()
			# print(folder+"\\"+line[0])
			check_img=cv2.imread(folder+"\\"+line[0],1)
			# print(check_img.shape)

			x = map(float, line[5::2])
			y = map(float, line[6::2])
			landmarks = list(zip(x, y))
			# print(len(landmarks))
			for i in range(len(landmarks)):
				# print(landmarks[i][0])
				face_key=cv2.circle(check_img,(int(landmarks[i][0]),int(landmarks[i][1])),1,(0,0,255),1)


			plt.imshow(cv2.cvtColor(face_key,cv2.COLOR_BGR2RGB))
			plt.show()


def main():
	data_images=load_images()
	new_name=crop_image(data_images)
	save_train_txt(data_images,new_name)
	save_test_txt(data_images,new_name)
	check_data(data_type = "train.txt",start_img=40,end_img=50)

if __name__=="__main__":
	main()
