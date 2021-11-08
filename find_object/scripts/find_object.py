#! /usr/bin/env python
# -*- coding: utf-8 -*-
 
import rospy
import cv2
import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
 
class find_object:
    def __init__(self):
        
        #ノードの登録
        rospy.init_node('find_object', anonymous=True)
        #パラメータの読み込み(実行時の引数を読み込み)
        self.temp_path = rospy.get_param('~path_name')
        
        self.bridge = CvBridge()
        #画像トピックを購読して指定したコールバック関数を実行
        rospy.Subscriber('camera/rgb/image_raw',Image, self.ImageCallback)
 
    def ImageCallback(self, img_data):
        try:
            #OpenCV等で扱える形式に変換
            img_ori = self.bridge.imgmsg_to_cv2(img_data, 'passthrough')
            #画像サイズを半分に縮小
            img = cv2.resize(img_ori, (int(img_ori.shape[1] * 0.5), int(img_ori.shape[0] * 0.5)))
            
        except CvBridgeError, e:
            rospy.logerr(e)
 
        #テンプレート画像を読み込み
        temp = cv2.imread(self.temp_path)

        # 色変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        temp2 = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

        #グレイスケールに変換した入力画像とテンプレート画像AKAZE記述子の計算
        detector = cv2.ORB_create()
        kp1, des1 = detector.detectAndCompute(temp2, None)
        kp2, des2 = detector.detectAndCompute(gray, None)

        #Knnによるマッチング
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        #マッチング結果を間引きし、良い結果のみを選定
        good = []
        ratio = 0.8
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        #マッチングによる対応点数がMIN_MARCH_NUM以上の場合にはホモグラフィー行列を計算
        #MIN_MARCH_NUMは大きい値ほど良いホモグラフィー行列が得られる
        MIN_MARCH_NUM = 5
        if len(good) >= MIN_MARCH_NUM:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h,w,c = temp.shape
            pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts,M)
    
            #テンプレート領域の描画
            img = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            #対応点を線で結ぶ
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

            img3 = cv2.drawMatches(temp,kp1,img,kp2,good,None,**draw_params)

            #結果画像の表示
            cv2.namedWindow("result")
            cv2.imshow("result", img3)
            cv2.waitKey(10)

        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MARCH_NUM)
            matchesMask = None

            cv2.namedWindow("result")
            cv2.imshow("result", img)
            cv2.waitKey(10)


if __name__ == '__main__':
    try:
        fo = find_object()
        rospy.spin()
    except rospy.ROSInterruptException: pass
