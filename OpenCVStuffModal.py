import bpy
from mathutils import *
import cv2 as cv
import numpy as np
from urllib.request import urlopen
import mediapipe as mp
import math
import time
import random



ob = bpy.data.objects['Miku_Armature']
#bpy.context.scene.objects.active = ob
#bpy.ops.object.mode_set(mode='POSE')

pbone = ob.pose.bones[ "Head" ]
pbone.rotation_mode = 'XYZ'

chest_bone = ob.pose.bones[ "Chest" ]
chest_bone.rotation_mode = 'XYZ'

RShoulder_bone = ob.pose.bones[ "Shoulder_R" ]
RShoulder_bone.rotation_mode = 'XYZ'

LShoulder_bone = ob.pose.bones[ "Shoulder_L" ]
LShoulder_bone.rotation_mode = 'XYZ'

L_Upper_bone = ob.pose.bones[ "L_Upper" ]
L_Upper_bone.rotation_mode = 'XYZ'

R_Upper_bone = ob.pose.bones[ "R_Upper" ]
R_Upper_bone.rotation_mode = 'XYZ'

L_Lower_bone = ob.pose.bones[ "L_Lower" ]
L_Lower_bone.rotation_mode = 'XYZ'

R_Lower_bone = ob.pose.bones[ "R_Lower" ]
R_Lower_bone.rotation_mode = 'XYZ'

Hip_bone = ob.pose.bones[ "Hip" ]
Hip_bone.rotation_mode = 'XYZ'


w, h = 360, 240
t_w , t_h = 640 , 480
font_size = 0.5
font_thick = 1
circle_size = 1
l = 15.8
smoothing_weight = 0.2
angular_offset = 0
distance = -1
phone1 = {"url": 'http://192.168.0.148:8080/shot.jpg', "angle": 63.1}
phone2 = {"url": 'http://192.168.0.196:6969/shot.jpg', "angle": 68.7}


cap = cv.VideoCapture(0)
#cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
#cap.set(cv2.CAP_PROP_FPS,30)


cap2 = cv.VideoCapture(1)
#cap2.set(cv2.CAP_PROP_BUFFERSIZE,3)
#cap2.set(cv2.CAP_PROP_FPS,30)


pose_distances = [-1 for _ in range(33)]
left_hand_distances = [ -1 for _ in range(21) ]

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
holistic2 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

Head_Z_Target = 0

def calcDistance(lm1, lm2, distance):
    Theta1 = math.radians((180 - (58.45 + (phone1["angle"] * lm1.x))))
    Theta2 = math.radians((180 - (55.65 + (phone2["angle"] * lm2.x))))

    Sin1 = math.sin(Theta2 - Theta1)
    Sin2 = math.sin(Theta1)

    if Sin1 != 0:
        Sin3 = (Sin2 / Sin1) * (-l)
        temp = Sin3 * math.sin(Theta2)
        temp -= angular_offset * ( ( (lm1.x + lm2.x) / 2 ) - 0.5 ) * 2
        if distance == -1:
            distance = temp
        else:
            to_add = temp * smoothing_weight
            distance *= 1 - smoothing_weight
            distance += to_add
    return distance


def getVector( index ):
    LS_lm1 = results1.pose_world_landmarks.landmark[11]
    LS_lm2 = results2.pose_world_landmarks.landmark[11]
    LS_X = ( (LS_lm1.x + LS_lm2.x) / 2 )
    LS_Y = ( (LS_lm1.y + LS_lm2.y) / 2 )
    LS_Z = ( (LS_lm1.z + LS_lm2.z) / 2 )
    return Vector( [ LS_X , LS_Z , LS_Y ] )


class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"
    
    max1 = 0
    min1 = -1
    
    
    Blink_Time = 0
    Blink_State = "Open"
    Cur_Blink_Pos = 1
    
    timer_count = 0

    _timer = None

    Head_Z_Target = 0
    Head_X_Target = 0
    
    Chest_Y_Target = 0
    Chest_Z_Target = 0
    
    L_Shoulder_X_Target = 0
    R_Shoulder_X_Target = 0
    
    L_Upper_X_Target = 0
    L_Upper_Z_Target = 0
    
    R_Upper_X_Target = 0
    R_Upper_Z_Target = 0
    
    L_Lower_X_Target = 0
    L_Lower_Z_Target = 0
    
    R_Lower_X_Target = 0
    R_Lower_Z_Target = 0
    
    

    def my_handler(self):
        
        t3 = time.time()
        if not ( cap.grab() and cap2.grab()):
            print("No more frames")
            return
        
        
        _, imgCv1 = cap.retrieve()
        _, imgCv2 = cap2.retrieve()
        t4 = time.time()
        #print( f"Read : {t4-t3}" )    
    
    
    
        t5 = time.time()    
        imgCv1 = cv.cvtColor(imgCv1, cv.COLOR_RGB2BGR)
        imgCv1 = cv.flip(imgCv1, 1)

#        imgNp2 = np.array(bytearray(imgResponse2.read()), dtype=np.uint8)
#        imgCv2 = cv.imdecode(imgNp2, -1)
        imgCv2 = cv.cvtColor(imgCv2, cv.COLOR_RGB2BGR)
        imgCv2 = cv.flip(imgCv2, 1)

        results1 = holistic1.process(imgCv1)
        results2 = holistic2.process(imgCv2)
        t6 = time.time()
        #print( f"Process : {t6-t5}" )


        t7 = time.time()
        if results1.pose_world_landmarks and results2.pose_world_landmarks:
            LS_lm1 = results1.pose_world_landmarks.landmark[11]
            LS_lm2 = results2.pose_world_landmarks.landmark[11]
            LS_X = ( (LS_lm1.x + LS_lm2.x) / 2 )
            LS_Y = ( (LS_lm1.y + LS_lm2.y) / 2 )
            LS_Z = ( (LS_lm1.z + LS_lm2.z) / 2 )
            L_Shoulder = Vector( [ LS_X , LS_Z , LS_Y ] )
            
            RS_lm1 = results1.pose_world_landmarks.landmark[12]
            RS_lm2 = results2.pose_world_landmarks.landmark[12]
            RS_X = ( (RS_lm1.x + RS_lm2.x) / 2 )
            RS_Y = ( (RS_lm1.y + RS_lm2.y) / 2 )
            RS_Z = ( (RS_lm1.z + RS_lm2.z) / 2 )
            R_Shoulder = Vector( [ RS_X , RS_Z , RS_Y ] )
            
            #---------------------------
            LMA_lm1 = results1.pose_world_landmarks.landmark[13]
            LMA_lm2 = results2.pose_world_landmarks.landmark[13]
            LMA_X = ( (LMA_lm1.x + LMA_lm2.x) / 2 )
            LMA_Y = ( (LMA_lm1.y + LMA_lm2.y) / 2 )
            LMA_Z = ( (LMA_lm1.z + LMA_lm2.z) / 2 )
            L_MA = Vector( [ LMA_X , LMA_Z , LMA_Y ] )
            
            RMA_lm1 = results1.pose_world_landmarks.landmark[14]
            RMA_lm2 = results2.pose_world_landmarks.landmark[14]
            RMA_X = ( (RMA_lm1.x + RMA_lm2.x) / 2 )
            RMA_Y = ( (RMA_lm1.y + RMA_lm2.y) / 2 )
            RMA_Z = ( (RMA_lm1.z + RMA_lm2.z) / 2 )
            R_MA = Vector( [ RMA_X , RMA_Z , RMA_Y ] )
            
            #---------------------------
            LLA_lm1 = results1.pose_world_landmarks.landmark[15]
            LLA_lm2 = results2.pose_world_landmarks.landmark[15]
            LLA_X = ( (LLA_lm1.x + LLA_lm2.x) / 2 )
            LLA_Y = ( (LLA_lm1.y + LLA_lm2.y) / 2 )
            LLA_Z = ( (LLA_lm1.z + LLA_lm2.z) / 2 )
            L_LA = Vector( [ LLA_X , LLA_Z , LLA_Y ] )
            
            RLA_lm1 = results1.pose_world_landmarks.landmark[16]
            RLA_lm2 = results2.pose_world_landmarks.landmark[16]
            RLA_X = ( (RLA_lm1.x + RLA_lm2.x) / 2 )
            RLA_Y = ( (RLA_lm1.y + RLA_lm2.y) / 2 )
            RLA_Z = ( (RLA_lm1.z + RLA_lm2.z) / 2 )
            R_LA = Vector( [ RLA_X , RLA_Z , RLA_Y ] )
            
            R_Upper_Arm_Vect = R_MA - R_Shoulder
            R_Upper_Arm_X = math.degrees(R_Upper_Arm_Vect.angle( Vector( [ 1 , 0 , 0 ] ) ) )
            R_Upper_Arm_Y = math.degrees(R_Upper_Arm_Vect.angle( Vector( [ 0 , 1 , 0 ] ) ) )
            R_Upper_Arm_Z = math.degrees(R_Upper_Arm_Vect.angle( Vector( [ 0 , 0 , 1 ] ) ) )
            
            
            L_Upper_Arm_Vect = L_MA - L_Shoulder
            L_Upper_Arm_X = math.degrees(L_Upper_Arm_Vect.angle( Vector( [ 1 , 0 , 0 ] ) ) )
            L_Upper_Arm_Y = math.degrees(L_Upper_Arm_Vect.angle( Vector( [ 0 , 1 , 0 ] ) ) )
            L_Upper_Arm_Z = math.degrees(L_Upper_Arm_Vect.angle( Vector( [ 0 , 0 , 1 ] ) ) )
            
            R_Lower_Arm_Vect = R_LA - R_MA
            R_Lower_Arm_X = math.degrees(R_Lower_Arm_Vect.angle( Vector( [ 1 , 0 , 0 ] ) ) )
            R_Lower_Arm_Y = math.degrees(R_Lower_Arm_Vect.angle( Vector( [ 0 , 1 , 0 ] ) ) )
            R_Lower_Arm_Z = math.degrees(R_Lower_Arm_Vect.angle( Vector( [ 0 , 0 , 1 ] ) ) )
            
            
            L_Lower_Arm_Vect = L_LA - L_MA
            L_Lower_Arm_X = math.degrees(L_Lower_Arm_Vect.angle( Vector( [ 1 , 0 , 0 ] ) ) )
            L_Lower_Arm_Y = math.degrees(L_Lower_Arm_Vect.angle( Vector( [ 0 , 1 , 0 ] ) ) )
            L_Lower_Arm_Z = math.degrees(L_Lower_Arm_Vect.angle( Vector( [ 0 , 0 , 1 ] ) ) )        
            
            #print( int(L_Upper_Arm_X) , int(L_Lower_Arm_X) , int(L_Upper_Arm_X)-int(L_Lower_Arm_X) )
            
            self.R_Upper_X_Target = 0.5-( 1.7 * (((int(R_Upper_Arm_Z)-10)/150) ) )
            self.R_Upper_Z_Target = 0.0-( 1.39 * (  (int(R_Upper_Arm_Y)-80)  ) / 60 )
            
#            R_Lower_Arm_X -= int(R_Upper_Arm_Z)
#            R_Upper_Arm_Y -= int(R_Upper_Arm_Y)
            
            self.R_Lower_Z_Target = 0.2-( 3* ( 1-(( int(R_Lower_Arm_X))/180) ) )# - self.R_Upper_Z_Target
            self.R_Lower_Y_Target = 0.0+( 1.43 * (  (int(R_Lower_Arm_Y)-80)  ) / 45 )
            
            #print( int(R_Upper_Arm_X-R_Lower_Arm_X) , int(R_Upper_Arm_Y-R_Lower_Arm_Y) , int(R_Upper_Arm_Z-R_Lower_Arm_Z)  )
            
            self.L_Upper_X_Target = -0.53+( 1.73 * ( ((int(L_Upper_Arm_Z)-5)/160) ) )
            self.L_Upper_Z_Target = 0.1-( 1.4 * (  (int(L_Upper_Arm_Y)-40)  ) / 120 )
            
            self.L_Lower_Z_Target = 0-( 2.5* ( (( int(L_Lower_Arm_X-20))/140) ) )# - self.L_Upper_Z_Target
            self.L_Lower_Y_Target = 0.0+( 1.43 * (  (int(L_Lower_Arm_Y)-80)  ) / 45 )
            
            
#            L_Upper_X_Target = 0
#            L_Upper_Z_Target = 0
#            
#            R_Upper_X_Target = 0
#            R_Upper_Z_Target = 0
#            
#            L_Lower_X_Target = 0
#            L_Lower_Z_Target = 0
#            
#            R_Lower_X_Target = 0
#            R_Lower_Z_Target = 0
    
            
            Shoulder_Vector = R_Shoulder - L_Shoulder
            
            Shoulder_X = math.degrees(Shoulder_Vector.angle( Vector([1,0,0]) ))
            Shoulder_Y = math.degrees(Shoulder_Vector.angle( Vector([0,1,0]) ))
            Shoulder_Z = math.degrees(Shoulder_Vector.angle( Vector([0,0,1]) ))
            
            self.Chest_Y_Target = ((Shoulder_Y-103)/60)*1.5
            
            
            Shoulder_Z_Angle = ( (int(Shoulder_Z)-88)/18 )*0.35
            self.L_Shoulder_X_Target = -1 * Shoulder_Z_Angle
            self.R_Shoulder_X_Target = -1 * Shoulder_Z_Angle
            
            #---------------------------
            HL_lm1 = results1.pose_world_landmarks.landmark[7]
            HL_lm2 = results2.pose_world_landmarks.landmark[7]
            HL_X = ( (HL_lm1.x + HL_lm2.x) / 2 )
            HL_Y = ( (HL_lm1.y + HL_lm2.y) / 2 )
            HL_Z = ( (HL_lm1.z + HL_lm2.z) / 2 )
            L_Head = Vector( [ HL_X , HL_Z , HL_Y ] )
            
            HR_lm1 = results1.pose_world_landmarks.landmark[8]
            HR_lm2 = results2.pose_world_landmarks.landmark[8]
            HR_X = ( (HR_lm1.x + HR_lm2.x) / 2 )
            HR_Y = ( (HR_lm1.y + HR_lm2.y) / 2 )
            HR_Z = ( (HR_lm1.z + HR_lm2.z) / 2 )
            R_Head = Vector( [ HR_X , HR_Z , HR_Y ] )
            
            Head_Yaw = R_Head - L_Head
            
            Head_Yaw_X = math.degrees(Head_Yaw.angle( Vector([1,0,0]) ))
            Head_Yaw_Y = math.degrees(Head_Yaw.angle( Vector([0,1,0]) ))
            Head_Yaw_Z = math.degrees(Head_Yaw.angle( Vector([0,0,1]) ))
            
            self.Head_Y_Target = 1.6*((( int(Head_Yaw_Y) - 70 )/70)-0.5)
            
            #print( int(Head_Yaw_X), int(Head_Yaw_Y) , int(Head_Yaw_Z) , self.Head_Y_Target)
            
            #---------------------------
            
            Neck_Vect = ( L_Shoulder + R_Shoulder )/2
            
            Neck_Vect_X = math.degrees(Shoulder_Vector.angle( Vector([1,0,0]) ))
            Neck_Vect_Y = math.degrees(Shoulder_Vector.angle( Vector([0,1,0]) ))
            Neck_Vect_Z = math.degrees(Shoulder_Vector.angle( Vector([0,0,1]) ))
            
            #print( [ int(Neck_Vect_X) , int(Neck_Vect_Y) , int(Neck_Vect_Z) ] )

            self.Chest_Z_Target = -(((int(Neck_Vect_Z))-90)/15)*0.2
            
            Face_lm1 = results1.pose_world_landmarks.landmark[0]
            Face_lm2 = results2.pose_world_landmarks.landmark[0]
            
            Face_L_lm1 = results1.pose_world_landmarks.landmark[8]
            Face_L_lm2 = results2.pose_world_landmarks.landmark[8]
            
            Face_R_lm1 = results1.pose_world_landmarks.landmark[7]
            Face_R_lm2 = results2.pose_world_landmarks.landmark[7]
            
            Face_X = ( (Face_lm1.x + Face_lm2.x + Face_L_lm1.x + Face_L_lm2.x + Face_R_lm1.x + Face_R_lm2.x) / 6 )
            Face_Y = ( (Face_lm1.y + Face_lm2.y + Face_L_lm1.y + Face_L_lm2.y + Face_R_lm1.y + Face_R_lm2.y) / 6 )
            Face_Z = ( (Face_lm1.z + Face_lm2.z + Face_L_lm1.z + Face_L_lm2.z + Face_R_lm1.z + Face_R_lm2.z) / 6 )
            Face_Vect = Vector( [ Face_X , Face_Z , Face_Y ] )
            
            Face_Angle_Vect = Face_Vect-Neck_Vect
                
            
            X = Face_Angle_Vect.angle( Vector([1,0,0]) )
            Y = Face_Angle_Vect.angle( Vector([0,1,0]) )
            Z = Face_Angle_Vect.angle( Vector([0,0,1]) )
            
            text = "Fail"
#            if Face_Angle_Vect.length > 0:        
#                print ( [ f"X : {X} " , f"Y : {Y} " , f"Z : {Z}" ] )
                            
            #pbone.rotation_euler.rotate_axis( "X" , Y )
            #pbone.rotation_euler.rotate_axis( "Y" , (Y))
            self.Head_Z_Target = -0.8 + ( 1.6 * (X-1.11)/1.048 )
            self.Head_X_Target = -0.0 +  ( 1.4 * (Y-2.139)/0.671 )
            if X > self.max1:
                self.max1 = X
            if X<self.min1 or self.min1 == -1:
                self.min1 = X
                
            RLA_lm1 = results1.pose_world_landmarks.landmark[23]
            RLA_lm2 = results2.pose_world_landmarks.landmark[23]
            RLA_X = ( (RLA_lm1.x + RLA_lm2.x) / 2 )
            RLA_Y = ( (RLA_lm1.y + RLA_lm2.y) / 2 )
            RLA_Z = ( (RLA_lm1.z + RLA_lm2.z) / 2 )
            Hip_L = Vector( [ RLA_X , RLA_Z , RLA_Y ] )
            
            RLA_lm1 = results1.pose_world_landmarks.landmark[24]
            RLA_lm2 = results2.pose_world_landmarks.landmark[24]
            RLA_X = ( (RLA_lm1.x + RLA_lm2.x) / 2 )
            RLA_Y = ( (RLA_lm1.y + RLA_lm2.y) / 2 )
            RLA_Z = ( (RLA_lm1.z + RLA_lm2.z) / 2 )
            Hip_R = Vector( [ RLA_X , RLA_Z , RLA_Y ] )
                
            Hip_Yaw = Hip_L - Hip_R
            
            Hip_X = math.degrees(Hip_Yaw.angle( Vector([1,0,0]) ))
            Hip_Y = math.degrees(Hip_Yaw.angle( Vector([0,1,0]) ))
            Hip_Z = math.degrees(Hip_Yaw.angle( Vector([0,0,1]) ))
            
            self.Hip_Y_Target = (2.8*(Hip_Y/180))-1.4
            
            print( int(Hip_X), int(Hip_Y) , int(Hip_Z) )
            
            
            #print( round(self.max1,3) , round(self.min1,3) )
            
        t8 = time.time()
        #print( f"Final : {t8-t7}" )



    def update_view(self):
        pbone.rotation_euler.z = (pbone.rotation_euler.z*0.8) + (self.Head_Z_Target*0.2)
        pbone.rotation_euler.x = (pbone.rotation_euler.x*0.8) + (self.Head_X_Target*0.2)
        
        chest_bone.rotation_euler.y = (chest_bone.rotation_euler.y*0.8) + (self.Chest_Y_Target*0.2)
        chest_bone.rotation_euler.z = (chest_bone.rotation_euler.z*0.8) + (self.Chest_Z_Target*0.2)
        
        RShoulder_bone.rotation_euler.x = (RShoulder_bone.rotation_euler.x*0.8) + (self.R_Shoulder_X_Target*0.2)
        LShoulder_bone.rotation_euler.x = (LShoulder_bone.rotation_euler.x*0.8) + (self.L_Shoulder_X_Target*0.2)
        
        L_Upper_bone.rotation_euler.x = (L_Upper_bone.rotation_euler.x*0.8) + (self.L_Upper_X_Target*0.2)
        L_Upper_bone.rotation_euler.z = (L_Upper_bone.rotation_euler.z*0.8) + (self.L_Upper_Z_Target*0.2)
        
        L_Lower_bone.rotation_euler.x = (L_Lower_bone.rotation_euler.x*0.8) + (self.L_Lower_X_Target*0.2)
        L_Lower_bone.rotation_euler.z = (L_Lower_bone.rotation_euler.z*0.8) + (self.L_Lower_Z_Target*0.2)
        
        R_Upper_bone.rotation_euler.x = (R_Upper_bone.rotation_euler.x*0.8) + (self.R_Upper_X_Target*0.2)
        R_Upper_bone.rotation_euler.z = (R_Upper_bone.rotation_euler.z*0.8) + (self.R_Upper_Z_Target*0.2)
        
        R_Lower_bone.rotation_euler.x = (R_Lower_bone.rotation_euler.x*0.8) + (self.R_Lower_X_Target*0.2)
        R_Lower_bone.rotation_euler.z = (R_Lower_bone.rotation_euler.z*0.8) + (self.R_Lower_Z_Target*0.2)
        
        Hip_bone.rotation_euler.y = (Hip_bone.rotation_euler.z*0.8) + (self.Hip_Y_Target*0.2)
        
        
    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            try:
                self.timer_count += 1
                if self.Blink_Time == 0:
                    self.Blink_Time = -1
                    self.Blink_State = "Closing"
                if self.Blink_State == "Closing":
                    self.Cur_Blink_Pos =  self.Cur_Blink_Pos*0.25 + 0*0.75
                if self.Blink_State == "Opening":
                    self.Cur_Blink_Pos =  self.Cur_Blink_Pos*0.4 + 1*0.6
                if self.Cur_Blink_Pos <= 0.1 and self.Blink_State == "Closing":
                    self.Cur_Blink_Pos = 0
                    self.Blink_State = "Opening"
                elif self.Cur_Blink_Pos >= 0.9 and self.Blink_State == "Opening": 
                    self.Cur_Blink_Pos = 1
                    self.Blink_State = "Open"
                    self.Blink_Time = random.randrange( 50 , 150 )
                self.Blink_Time -= 1
                bpy.data.shape_keys["Key"].key_blocks["Eyes_Close"].value = 1 - self.Cur_Blink_Pos

                    
                if self.timer_count%5 == 0:
                    self.my_handler()
                self.update_view()           
            except Exception as e:
                print( e )
            # change theme color, silly!
        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        #bpy.app.handlers.frame_change_pre.append(update_view)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cap.release()
        cap2.release()
        print("canceled")
        #bpy.app.handlers.frame_change_pre.remove(update_view)


def register():
    bpy.utils.register_class(ModalTimerOperator)


def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.modal_timer_operator()
