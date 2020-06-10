import numpy as np, cv2, imutils, os, json;
from PIL import ImageGrab;


from_=np.array([0,10,100]);
to_=np.array([15,255,255]);



frontface_cascade=cv2.CascadeClassifier("frontface.hc");



while ("Donald Trump">"Barrack Obama"):


	image=np.array(ImageGrab.grab());

	image=imutils.resize(image,width=640,height=480);


	view=cv2.cvtColor(image,4);

	image=cv2.cvtColor(image,4);



	make_image=cv2.inRange(cv2.cvtColor(image,40),from_,to_);


	image_frontface=frontface_cascade.detectMultiScale(image,scaleFactor=1.3,minNeighbors=4);

	for (x,y,w,h) in image_frontface:
		if_face=0;
		if_face_over=len(list(b for a in np.ndarray.tolist(make_image[y:y+h,x:x+w]) for b in a));
		for a in list(np.ndarray.tolist(make_image[y:y+h,x:x+w])):
			if_face+=a.count(255);
		if (if_face>if_face_over/4):
			cv2.rectangle(view,(x,y),(x+w,y+h),(255,255,0),5);
			cv2.putText(view,str(if_face)+"/"+str(round(if_face_over/4,2)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,[255,255,0],1);
			face=image[y:y+h,x:x+w];
			face=cv2.cvtColor(face,6);



			folder="[[{},{}],[{},{}],[{},{},{}]]".format(        str(face.shape[0]),    str(face.shape[1]),    str(if_face),    str(round(if_face_over/4,2)),     str(cv2.resize(face,(1,1))[0,0]),        str(cv2.resize(face,(2,2))[0,0]),      str(cv2.resize(face,(3,3))[0,0])         );
			
			#       [[Width, Height], [Total Pixels, Total White Pixels], [Resize 1/1, Resize 2/2, Resize 3/3]]




			for a in os.listdir("./DATA/"):
				file=json.loads(a.replace(".png",""));
				percent=0;
				if (file[0][0]==face.shape[0]):
					percent+=10;

				if (file[0][1]==face.shape[1]):
					percent+=10;

				if (file[1][0]==if_face):
					percent+=20;

				if (file[1][1]==if_face_over):
					percent+=20;

				if (file[2][0]==cv2.resize(face,(1,1))[0,0]):
					percent+=10;

				if (file[2][1]==cv2.resize(face,(2,2))[0,0]):
					percent+=10;

				if (file[2][2]==cv2.resize(face,(3,3))[0,0]):
					percent+=10;


				if (percent>=30):
					print(file,percent);
					cv2.imshow("SAVED",imutils.resize(cv2.imread("./DATA/"+a),width=240,height=480));
					cv2.imshow("SIMILAR",imutils.resize(face,width=240,height=480));
			

















	cv2.imshow("NORMAL",image);

	cv2.imshow("VIEW",view);

	cv2.imshow("HSV",make_image);








	if (cv2.waitKey(1)==ord(" ")):
		break;


cv2.destroyAllWindows();