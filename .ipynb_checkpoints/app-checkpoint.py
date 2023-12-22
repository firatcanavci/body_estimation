import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Calculate Angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# Calculate Distance
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance





# CSS stilini tanımla
style = """
<style>
.reportview-container .main .block-container{
    background-color: #ffffe0;  /* Açık sarı arka plan rengi */
}
.sidebar .sidebar-content {
    background-color: #d3d3d3;  /* Açık gri sidebar rengi */
}
</style>
"""



# Sidebar başlığı ve grafik seçimi için checkboxlar
st.sidebar.title('Exercises')

# Bridge Position GIF
bridge_gif_link = "https://herbands.com/cdn/shop/files/bridge_e3ad6939-f9fc-453c-81a9-5f2758abdc06_540x.gif?v=1614342939"  # Yerine kendi bağlantınızı ekleyin
st.sidebar.markdown(f'<img src="{bridge_gif_link}" alt="Bridge Position" width="100%" style="max-width: 200px; margin-bottom: 10px;">', unsafe_allow_html=True)
show_bridge_position = st.sidebar.checkbox('Bridge Position')



# Jumping Jack GIF
jumping_jack_gif_link = "https://i.pinimg.com/originals/39/31/b8/3931b8eded1e338e2a4cb34722195bcb.gif"
st.sidebar.markdown(f'<img src="{jumping_jack_gif_link}" alt="Jumping Jacks" width="100%" style="max-width: 200px; margin-bottom: 10px;">', unsafe_allow_html=True)
show_jumping_jack = st.sidebar.checkbox('Jumping Jack Count')


# Plank Position GIF
plank_gif_link = "https://www.fitstream.com/images/bodyweight-training/bodyweight-exercises/plank.png"  # Yerine kendi bağlantınızı ekleyin
st.sidebar.markdown(f'<img src="{plank_gif_link}" alt="Plank Position" width="100%" style="max-width: 200px; margin-bottom: 10px;">', unsafe_allow_html=True)
show_plank_position = st.sidebar.checkbox('Plank Position')


if show_bridge_position:
    # Stili Streamlit uygulamasına ekle
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 22px 6px 8px rgba(150, 150, 150, 0.2);">Bridge Position</p>', unsafe_allow_html=True)
    # Kullanıcıdan video dosyası yükleme
    uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])

    

    
    # Kullanıcıdan video dosyası yükleme ve işleme
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_file = tfile.name
    
        cap = cv2.VideoCapture(video_file)
    
        cap.set(3, 640)  # Genişlik
        cap.set(4, 480) 
        st_video_2 = st.empty()
        # Curl counter variables
        counter = 0 
        stage = None
        
        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break  # Break the loop if there are no more frames
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
              
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    
                    #Calculate angle
                    angle_hip = calculate_angle(shoulder, hip, knee)
                    
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    angle_knee = calculate_angle(hip, knee, ankle)
                    
                    # Get coordinates Distance
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    distance_knee = calculate_distance(right_knee, left_knee)
                    
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    distance_shoulder = calculate_distance(right_shoulder, left_shoulder)
                    
                    # Visualize angle
                    # Setup status box in the top-right corner
                    status_box_position = (image.shape[1] - 225, 0)
                    status_box_size = (225, 90)
        
                    # Draw a rectangle in the top-right corner
                    cv2.rectangle(image, status_box_position, tuple(np.add(status_box_position, status_box_size)), (245, 117, 16), -1)
        
                    # Put text inside the rectangle
                    cv2.putText(image, f"AngleHip: {angle_hip:.2f}", 
                                tuple(np.add(status_box_position, (10, 20))),  # Adjust the position inside the rectangle
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                    )
        
                    cv2.putText(image, f"AngleKnee: {angle_knee:.2f}", 
                                tuple(np.add(status_box_position, (10, 40))),  # Adjust the position inside the rectangle
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA
                    )
        
                    cv2.putText(image, f"Distance_knee: {distance_knee:.2f}", 
                                tuple(np.add(status_box_position, (10, 60))),  # Adjust the position inside the rectangle
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                    )
        
                    cv2.putText(image, f"Distance_shoulder: {distance_shoulder:.2f}", 
                                tuple(np.add(status_box_position, (10, 80))),  # Adjust the position inside the rectangle
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA
                    )
        
                    # Draw blue circles at specific locations
                    width, height = cap.get(3), cap.get(4)
                    cv2.circle(image, (int(hip[0] * width), int(hip[1] * height)), 20, (0, 255, 255), -1)
                    cv2.circle(image, (int(knee[0] * width), int(knee[1] * height)), 20, (0, 165, 255), -1)
        
                    if (distance_knee < distance_shoulder) and (distance_knee > 0.01) and (angle_knee < 85.0):
                        # Draw a circle below the rectangle
                        circle_center = (status_box_position[0] + status_box_size[0] // 2, status_box_position[1] + status_box_size[1] + 40)
                        cv2.circle(image, circle_center, 30, (0, 255, 0), -1)
        
                        # Curl counter logic
                        if angle_hip > 170:
                            stage = "up"
                        if angle_hip < 135 and stage == 'up':
                            stage = "down"
                            counter += 1
                            print(counter)
                    else:
                        cv2.circle(image, circle_center, 30, (0, 0, 255), -1)
                except:
                    pass
        
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        
                # Rep data
                cv2.putText(image, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
                # Stage data
                cv2.putText(image, 'STAGE', (65, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
               
                # Pose estimation sonuçlarını içeren frame'i Streamlit'te göster
                st_video_2.video(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
            cap.release()







if show_jumping_jack:
    # Stili Streamlit uygulamasına ekle
    st.markdown(style, unsafe_allow_html=True)
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 22px 6px 8px rgba(150, 150, 150, 0.2);">Jumping Jack</p>', unsafe_allow_html=True)
    # Kullanıcıdan video dosyası yükleme
    uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
    
    # Kullanıcıdan video dosyası yükleme ve işleme
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_file = tfile.name
    
        cap = cv2.VideoCapture(video_file)
    
        # st.empty() ile boş bir widget oluştur
        st_video_1 = st.empty()
    
        counter_1 = 0
        stage_1 = None
    
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    
        #Pose estimation ve jumping jack sayma algoritmanız burada yer alacak
        # Açı hesaplama fonksiyonu
        def calculate_angle(a, b, c):
            a = np.array(a)  
            b = np.array(b)  
            c = np.array(c)  
    
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
    
            if angle > 180.0:
                angle = 360-angle
    
            return angle
    
    
        # Videodaki frame'leri işle ve göster
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
    
                # Pose estimation işlemi
                results = pose.process(image)
    
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
                try:
                    landmarks = results.pose_landmarks.landmark
    
    
                    # Sağ ve sol omuz koordinatları
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    
                    # Sağ ve sol bilek ve kalça koordinatları
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    
                    # Açıları hesapla
                    angle_r = calculate_angle(r_wrist, r_shoulder, r_hip)
                    angle_l = calculate_angle(l_wrist, l_shoulder, l_hip)
    
                    # Jumping jack sayacı mantığı
                    if angle_r > 160 and angle_l > 160:
                        stage_1 = "up"
                    elif angle_r < 160 and angle_l < 160 and stage_1 == "up":
                        stage_1 = "down"
                        counter_1 += 1
                            
                except:
                    pass
    
                # REPS metin kutusunun konumunu sağ üst köşeye ayarla
                frame_width = frame.shape[1]
                box_width = 225
                box_height = 73
                cv2.rectangle(image, (frame_width - box_width, 0), (frame_width, box_height), (245,117,16), -1)
                
                # Rep ve Stage verilerini yazdır
                cv2.putText(image, 'REPS', (frame_width - box_width + 15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_1), 
                            (frame_width - box_width + 10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Algılamaları göster
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    
                # Pose estimation sonuçlarını içeren frame'i Streamlit'te göster
                st_video_1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
            cap.release()




if show_plank_position:
    # Stili Streamlit uygulamasına ekle
    st.markdown(style, unsafe_allow_html=True)
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 22px 6px 8px rgba(150, 150, 150, 0.2);">Plank Position</p>', unsafe_allow_html=True)
    # Kullanıcıdan video dosyası yükleme
    uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # st.empty() ile boş bir widget oluştur
    st_video_3 = st.empty()
    # Initialize a counter variable and a start time variable
    counter_3 = 0
    start_time = time.time()
    elapsed_time = 0 
        
    # Kullanıcıdan video dosyası yükleme ve işleme
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_file = tfile.name
    
        cap = cv2.VideoCapture(video_file)
    
        def calculate_angle(a, b, c):
            a = np.array(a)  
            b = np.array(b)  
            c = np.array(c)  
    
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
    
            if angle > 180.0:
                angle = 360-angle
    
            return angle     

        
        # Video çözünürlüğünü alın
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
        
                
                if not ret:
                    # Break the loop if there are no more frames to read
                    break
                    

  

                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
              
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[11].x,landmarks[11].y]
                    hip = [landmarks[23].x,landmarks[23].y]
                    knee = [landmarks[25].x,landmarks[25].y]
                    elbow = [landmarks[13].x,landmarks[13].y]
                    wrist = [landmarks[15].x,landmarks[15].y]
                    #angle_hip = calculate_angle(shoulder, hip, knee)
                    #angle_elbow = calculate_angle(shoulder, elbow, wrist)
                    # Calculate angles
                    angle_hip = calculate_angle(shoulder, hip, knee)
                    angle_elbow = calculate_angle(shoulder, elbow, wrist)
                    
                    
                    # Görüntü boyutlarını alma
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]
                    
                    # Daire parametreleri
                    radius = 30
                    
                    # Sağ üst köşe koordinatları
                    top_right_x = frame_width - 1
                    top_right_y = 0
                    
                    # Daire merkez koordinatları
                    circle_center = (top_right_x - radius, top_right_y + radius)         
                    
                    # Check if both angle_hip and angle_elbow meet the conditions
                    if 140 <= angle_hip <= 160 and 70 <= angle_elbow <= 85:
                        counter_3 += 1
                        if start_time is None:
                            start_time = time.time()
            
                        # Draw a green circle in the top-right corner when conditions are met
                        cv2.circle(image, (1130, 50), 30, (0, 255, 0), -1)
                    else:
                        if start_time is not None:
                            elapsed_time += time.time() - start_time
                            start_time = None
            
                        # Draw a red circle in the top-right corner when conditions are not met
                        cv2.circle(image, (1130, 50), 30, (0, 0, 255), -1)
            
                    # Display angle_hip, angle_elbow, and counter values on the video
                    cv2.putText(image, f"Angle Hip: {angle_hip:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Angle Elbow: {angle_elbow:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
                    # Draw blue circles at specific locations
                    cv2.circle(image, (int(hip[0] * width), int(hip[1] * height)), 20, (0, 255, 255), -1)
                    cv2.circle(image, (int(elbow[0] * width), int(elbow[1] * height)), 20, (0, 165, 255), -1)
                    
                    
                    cv2.circle(image, (300, 25), 15, (0, 255, 255), -1)
                    cv2.circle(image, (300, 55), 15, (0, 165, 255), -1)
                
                except:
                    pass
            
                if start_time is not None:
                    elapsed_time += time.time() - start_time
                    start_time = time.time()
            
                # Display elapsed_time as counter (sağ üst köşe, saniye cinsinden)
                cv2.putText(image, f"Elapsed Time: {elapsed_time:.2f} seconds", (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)) 
                                # Pose estimation sonuçlarını içeren frame'i Streamlit'te göster
                st_video_3.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
            cap.release()