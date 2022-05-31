# Written by Lex Whalen

import cv2 as cv
import numpy as np
import os
import math
from frame_operations import FrameOperations

class PoseEstimator():

    def __init__(self):
        self.FRAME_OPS = FrameOperations()


        self.BODY_PARTS =  { "Nose": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "RightWrist": 4,
               "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9,
               "RightAnkle": 10, "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "RightEye": 14,
               "LeftEye": 15, "RightEar": 16, "LeftEar": 17, "Background": 18 }
            
        self.POSE_PAIRS = [ ["Neck", "RightShoulder"], ["Neck", "LeftShoulder"], ["RightShoulder", "RightElbow"],
               ["RightElbow", "RightWrist"], ["LeftShoulder", "LeftElbow"], ["LeftElbow", "LeftWrist"],
               ["Neck", "RightHip"], ["RightHip", "RightKnee"], ["RightKnee", "RightAnkle"], ["Neck", "LeftHip"],
               ["LeftHip", "LeftKnee"], ["LeftKnee", "LeftAnkle"], ["Neck", "Nose"], ["Nose", "RightEye"],
               ["RightEye", "RightEar"], ["Nose", "LeftEye"], ["LeftEye", "LeftEar"] ]

        self.CWD = os.getcwd()
        self.RESOURCES = os.path.join(self.CWD,'resources')
        self.GRAPH_OPT = os.path.join(self.RESOURCES,'graph_opt.pb')

        self.NET = cv.dnn.readNetFromTensorflow(self.GRAPH_OPT)
        self.THR = 0.1
        self.IN_WIDTH = 396
        self.IN_HEIGHT = 368

        self.POINTS = []

        self.KEY_DISTANCES = {"RightArm":{"RightShoulder-RightElbow":None,"RightElbow-RightWrist":None,"Neck-RightShoulder":None},
        "LeftArm":{"LeftShoulder-LeftElbow":None,"LeftElbow-LeftWrist":None,"Neck-LeftShoulder":None},
        "RightLeg":{"RightHip-RightKnee":None,"RightKnee-RightAnkle":None},
        "LeftLeg":{"LeftHip-RightKnee":None,"LeftKnee-RightAnkle":None}}

        self.KEY_ANGLES = {"RightArm": [],"LeftArm":[],"RightLeg":[],"LeftLeg":[]}

        self.TEXT_COLOR = (0,0,0)

    def radianToDegree(self,radian):
        return radian * (180/math.pi)
   

    def get_pose_key_angles(self, frame, wantBlank = False):
        """applies pose estimation on frame, gets the distances between points"""

        # for the key points that do not come in pairs
        RightShoulder_pos = None
        RightWrist_pos = None

        LeftShoulder_pos = None
        LeftWrist_pos = None

        Neck_pos = None
        
        RightElbow_pos = None
        LeftElbow_pos = None

        RightHip_pos = None
        RightKnee_pos = None
        RightAnkle_pos = None

        LeftHip_pos = None
        LeftKnee_pos = None
        LeftAnkle_pos = None


        frame_height,frame_widght = frame.shape[0:2]
            
        self.NET.setInput(cv.dnn.blobFromImage(frame, 1.0, (self.IN_WIDTH, self.IN_HEIGHT), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.NET.forward()

        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(self.BODY_PARTS) == out.shape[1])

        #cLeftEar to get new points
        self.POINTS.cLeftEar()

        for i in range(len(self.BODY_PARTS)):

            heatMap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatMap)
            widght = (frame_widght * point[0]) / out.shape[3]
            height = (frame_height * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            if(conf > self.THR):
                self.POINTS.append((int(widght),int(height)))
            else:
                self.POINTS.append(None)

        # create blank frame overlay once OpenPose has read original frame so as to work
        if wantBlank:

            frame = np.zeros((frame_height,frame_widght,3),np.uint8)

            self.TEXT_COLOR = (255,255,255)

        for pair in self.POSE_PAIRS:

            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)

            # continuing ex: idFrom = BODY_PART["Neck"] returns 1
            # similarly, idTo = BODY_PARTS["RightShoulder"] returns 2
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if self.POINTS[idFrom] and self.POINTS[idTo]:

                # we use law of cosines to find angle c: 
                # cos(C) = (a^2 + b^2 - c^2) / 2ab

                if(partFrom == "RightShoulder"):
                    RightShoulder_pos = self.POINTS[idFrom]

                if(partTo == "RightWrist"):
                    RightWrist_pos = self.POINTS[idTo]

                if(partFrom == "LeftShoulder"):
                    LeftShoulder_pos = self.POINTS[idFrom]

                if(partTo == "LeftWrist"):
                    LeftWrist_pos = self.POINTS[idTo]

                if(partFrom == "Neck"):
                    Neck_pos = self.POINTS[idFrom]
                
                if(partTo == "RightElbow"):
                    RightElbow_pos = self.POINTS[idTo]

                if(partTo == "LeftElbow"):
                    LeftElbow_pos = self.POINTS[idTo]

                if(partFrom == "RightHip"):
                    RightHip_pos = self.POINTS[idFrom]
                
                if(partTo == "RightKnee"):
                    RightKnee_pos = self.POINTS[idTo]
                
                if(partTo == "RightAnkle"):
                    RightAnkle_pos = self.POINTS[idTo]
                    
                if(partFrom == "LeftHip"):
                    LeftHip_pos = self.POINTS[idFrom]
                
                if(partTo == "LeftKnee"):
                    LeftKnee_pos = self.POINTS[idTo]
                
                if(partTo == "LeftAnkle"):
                    LeftAnkle_pos = self.POINTS[idTo]

                if(partFrom == "RightShoulder" and partTo == "RightElbow"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["RightArm"]["RightShoulder-RightElbow"] = dist_2

                elif(partFrom == "RightElbow" and partTo == "RightWrist"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["RightArm"]["RightElbow-RightWrist"] = dist_2

                elif(partFrom == "LeftShoulder" and partTo == "LeftElbow"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["LeftArm"]["LeftShoulder-LeftElbow"] = dist_2

                elif(partFrom == "LeftElbow" and partTo == "LeftWrist"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["LeftArm"]["LeftElbow-LeftWrist"] = dist_2

                elif(partFrom == "Neck" and partTo == "RightShoulder"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["RightArm"]["Neck-RightShoulder"] = dist_2

                elif(partFrom == "Neck" and partTo == "LeftShoulder"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["LeftArm"]["Neck-LeftShoulder"] = dist_2

                elif(partFrom == "RightHip" and partTo == "RightKnee"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["RightLeg"]["RightHip-RightKnee"] = dist_2

                elif(partFrom == "RightKnee" and partTo == "RightAnkle"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["RightLeg"]["RightKnee-RightAnkle"] = dist_2
                
                elif(partFrom == "LeftHip" and partTo == "LeftKnee"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["LeftLeg"]["LeftHip-LeftKnee"] = dist_2

                elif(partFrom == "LeftKnee" and partTo == "LeftAnkle"):
                    dist_2 = (self.POINTS[idFrom][0] - self.POINTS[idTo][0]) **2 + (self.POINTS[idFrom][1] - self.POINTS[idTo][1]) **2
                    self.KEY_DISTANCES["LeftLeg"]["LeftKnee-LeftAnkle"] = dist_2

                # check if you want to return just the blank, or the image with the angles.

                cv.line(frame, self.POINTS[idFrom], self.POINTS[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, self.POINTS[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, self.POINTS[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        if(RightShoulder_pos is not None and RightWrist_pos is not None):

            rightShoulderToWrist = (RightShoulder_pos[0] - RightWrist_pos[0])**2 + (RightShoulder_pos[1] - RightWrist_pos[1])**2

            rightShoulderToElbow = self.KEY_DISTANCES["RightArm"]["RightShoulder-RightElbow"]
            rightElbowToWrist = self.KEY_DISTANCES["RightArm"]["RightElbow-RightWrist"]

            try:
                theta = self.radianToDegree(math.acos((rightShoulderToElbow  + rightElbowToWrist - rightShoulderToWrist)/(2 * math.sqrt(rightShoulderToElbow * rightElbowToWrist))))

            except ZeroDivisionError:
                theta = "Error"

            self.KEY_ANGLES["RightArm"].append(theta)

            # display the angle at the center joint. Use self.BODY_PARTS to find joint indices
            
            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[3],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        if(LeftShoulder_pos is not None and LeftWrist_pos is not None):

            leftShoulderToWrist = (LeftShoulder_pos[0] - LeftWrist_pos[0])**2 + (LeftShoulder_pos[1] - LeftWrist_pos[1])**2

            leftShoulderToElbow = self.KEY_DISTANCES["LeftArm"]["LeftShoulder-LeftElbow"]
            leftElbowToWrist = self.KEY_DISTANCES["LeftArm"]["LeftElbow-LeftWrist"]

            try:
                theta = self.radianToDegree(math.acos((leftShoulderToElbow  + leftElbowToWrist - leftShoulderToWrist)/(2 * math.sqrt(leftShoulderToElbow * leftElbowToWrist))))

            except ZeroDivisionError:
                theta = None

            self.KEY_ANGLES["LeftArm"].append(theta)

            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[6],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if(Neck_pos is not None and LeftElbow_pos is not None):

            leftNeckToElbow = (Neck_pos[0] - LeftElbow_pos[0])**2 + (Neck_pos[1] - LeftElbow_pos[1])**2

            leftNeckToShoulder = self.KEY_DISTANCES["LeftArm"]["Neck-LeftShoulder"]
            leftShoulderToElbow = self.KEY_DISTANCES["LeftArm"]["LeftShoulder-LeftElbow"]

            try:
                theta = self.radianToDegree(math.acos((leftNeckToShoulder + leftShoulderToElbow - leftNeckToElbow)/(2 * math.sqrt(leftNeckToShoulder * leftShoulderToElbow))))

            except ZeroDivisionError:
                theta = None
                
            self.KEY_ANGLES["LeftArm"].append(theta)

            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[5],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if(Neck_pos is not None and RightElbow_pos is not None):

            rightNeckToElbow = (Neck_pos[0] - RightElbow_pos[0])**2 + (Neck_pos[1] - RightElbow_pos[1])**2

            rightNeckToShoulder = self.KEY_DISTANCES["RightArm"]["Neck-RightShoulder"]
            rightShoulderToElbow = self.KEY_DISTANCES["RightArm"]["RightShoulder-RightElbow"]

            try:
                theta = self.radianToDegree(math.acos((rightNeckToShoulder + rightShoulderToElbow - rightNeckToElbow)/(2 * math.sqrt(rightNeckToShoulder * rightShoulderToElbow))))

            except ZeroDivisionError:
                theta = None

            self.KEY_ANGLES["RightArm"].append(theta)

            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[2],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if(RightHip_pos is not None and RightAnkle_pos is not None):

            rightHipToAnkle = (RightHip_pos[0] - RightAnkle_pos[0])**2 + (RightHip_pos[1] - RightAnkle_pos[1])**2

            rightHipToKnee = self.KEY_DISTANCES["RightLeg"]["RightHip-RightKnee"]
            rightKneeToAnkle = self.KEY_DISTANCES["RightLeg"]["RightKnee-RightAnkle"]

            try:
                theta = self.radianToDegree(math.acos((rightHipToKnee + rightKneeToAnkle - rightHipToAnkle)/(2 * math.sqrt(rightHipToKnee * rightKneeToAnkle))))

            except ZeroDivisionError:
                theta = None

            self.KEY_ANGLES["RightLeg"].append(theta)

            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[9],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if(LeftHip_pos is not None and LeftAnkle_pos is not None):

            leftHipToAnkle = (LeftHip_pos[0] - LeftAnkle_pos[0])**2 + (LeftHip_pos[1] - LeftAnkle_pos[1])**2

            leftHipToKnee = self.KEY_DISTANCES["LeftLeg"]["LeftHip-LeftKnee"]
            leftKneeToAnkle = self.KEY_DISTANCES["LeftLeg"]["LeftKnee-LeftAnkle"]

            try:
                theta = self.radianToDegree(math.acos((leftHipToKnee + leftKneeToAnkle - leftHipToAnkle)/(2 * math.sqrt(leftHipToKnee * leftKneeToAnkle))))

            except ZeroDivisionError:
                theta = None

            self.KEY_ANGLES["LeftLeg"].append(theta)

            if(theta is not None):
                cv.putText(frame,"{:.1f}".format(theta),self.POINTS[12],cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        t, _ = self.NET.getPerfProfile()
        frequency = cv.getTickFrequency() / 1000

        cv.putText(frame, '%.2fms' % (t / frequency), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_COLOR)

        return frame
