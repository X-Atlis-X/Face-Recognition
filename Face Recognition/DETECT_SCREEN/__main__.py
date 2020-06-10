import numpy as np, cv2, imutils;
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
			cv2.imwrite("./FACES/{}{}{}{}.png".format(str(x),str(w),str(y),str(h)),face);
			make_face=make_image[y:y+h,x:x+w];
			face=imutils.resize(face,width=360,height=0);
			make_face=imutils.resize(make_face,width=360,height=0);
			cv2.imshow("FACE",face);
			cv2.imshow("HSV_FACE",make_face);







	cv2.imshow("NORMAL",image);

	cv2.imshow("VIEW",view);

	cv2.imshow("HSV",make_image);








	if (cv2.waitKey(1)==ord(" ")):
		break;


cv2.destroyAllWindows();