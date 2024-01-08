import cv2
import numpy as np

class Detector:
    def __init__(self, video_path, dp=1.11, minDist=600, param1=150, param2=41, minRadius=360, maxRadius=420, comp=[10,15], rho=1.32, theta=np.pi/180, threshold=115, min_line_len=100, max_line_gap=10):
        self.video_path = video_path
        self.dp = dp # wsp odwrotnej rozdzielczosci wideo do akumulatora. jezeli =1 to takei same rozdzielczosci. im wiekszy wsp tym szybsza ale mniej dokladana detekcj. 
        self.minDist = minDist # minimalny dystana pomiedzy kolejnymi znalezionymi okregami 
        self.param1 = param1 # parametr dla krawdzedzi. prog dla operatora canny'ego
        self.param2 = param2 # parametr dla srodka okregu. im wiekszy tym wiecej znajduje okregow 
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.found_circle = None
        self.comp = comp  # stała kompensujaca srodek tarczy dla mniejszych okregow
        self.rho = rho # odstep miedzy liniami w przestrzeni Houghesa. im mniejszy tym wiecej linii wykrywa 
        self.theta = theta # rrozdzielczosc katowa w przestrzeni houghesa. tak samo jak rho
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.first_frame_lines = None   # lista do przechowywania linii miedzy strefami stworzona na 1 klatce 
        self.dart_points_info = [] # do przechowywania informacji o lotkach (x,y) angle distance
        # dart_points_info ma miec maksymalnie 3 wartosci, czyli tyle ile masksymalnie lotek w tarczy 
        self.total_points = 0
        self.sepp_points = [] # punkty w zaleznosci od lotki [1lotka,2lotka,3lotka]
        # tu tez maks 3 lotki 

    # Szukanie największego okręgu na klatkach
# --------------------------------------------OKREGI---------------------------------------------------
    def findCircles(self, frame):
        # Konwersja na odcienie szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Usuwanie szumów 
        gray = cv2.medianBlur(gray, 5) # przyjmuje rozmiar okoliczy wokol ktorej stosuje filtr
        gray = cv2.GaussianBlur(gray, (5,5), 0)  # rozmiar, odchylenie standardowe. 0 oznacza samoistne dobranie wg roziaru okna 
        # Zastosowanie tych dwóch po sobie jest najlepsze, 
        #oddzielnie widac wciaz szumy
        
        # Wykrywanie krawędzi
        edges_frame = cv2.Canny(gray, 50, 150,None,3) # 50, 150 progi, 3 rozmair kernela Sobela

        # Wykrywanie okręgów
        circles = cv2.HoughCircles(edges_frame, cv2.HOUGH_GRADIENT, self.dp, minDist=self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
        # zwraca x,y srodka okregu oraz promien

        # usuwanie wiadomosci o innych okregach niz pierwszy. zaokraglanie wartoci do calkowitych 
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.found_circle = circles[0]
        
        return self.found_circle

    # Rysowanie okręgów na klatce
    def drawCircles(self, frame):
        if self.found_circle is not None:
            (x, y, r) = self.found_circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        return frame

    # Rysowanie okręgów wewnątrz wykrytego okręgu
    def drawInnerCircles(self, frame, comp):
        if self.found_circle is not None:
            # Parametry tarczy
            (x, y, r) = self.found_circle
            # Poszczególne mniejsze okręgi
            inner_r = int(r * 0.08)  
            inner_r2 = int(r * 0.03) 
            bigger_r = int(r * 0.47)
            bigger_r2 = int(r * 0.42)
            biggest_r = int(r * 0.75)
            biggest_r2 = int(r * 0.7)
            cv2.circle(frame, (x-comp[0], y+comp[1]), inner_r, (0, 0, 255), 2) 
            cv2.circle(frame, (x-comp[0], y+comp[1]), inner_r2, (0, 0, 255), 2)
            cv2.circle(frame, (x-int(comp[0]/2), y+int(comp[1]/2)), biggest_r, (0, 0, 255), 2)
            cv2.circle(frame, (x-int(comp[0]/2), y+int(comp[1]/2)), biggest_r2, (0, 0, 255), 2)
            cv2.circle(frame, (x-int(comp[0]/2), y+int(comp[1])), bigger_r, (0, 0, 255), 2)
            cv2.circle(frame, (x-int(comp[0]/2), y+int(comp[1])), bigger_r2, (0, 0, 255), 2)
        
        return frame

# --------------------------------------------LINIE---------------------------------------------------
    # Szukanie linii na klatce
    def findLines(self, frame):
        # Skala szarości
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Usuwanie szumów
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        # Operacje morfologiczne 
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=3)
        gray = cv2.erode(gray, kernel, iterations=3)
        # Wykrycie krawędzi
        edges_frame = cv2.Canny(gray, 50, 150,None,3)
        cv2.imshow('CannyLines',edges_frame)

        lines = cv2.HoughLinesP(edges_frame, self.rho, self.theta, self.threshold, np.array([]), minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)
        # propabilistyczna metoda zwraca tablice punktow ktore sa koncami znalezionych odcinkow 

        # Odfiltrowanie linii poza okręgiem
        if self.found_circle is not None and lines is not None:
            (x, y, r) = self.found_circle
            new_lines = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # wektory od srodka do linii musza byc mniejsze niz promien tarczy
                    if ((x1-x)**2 + (y1-y)**2 <= r**2) and ((x2-x)**2 + (y2-y)**2 <= r**2):
                        new_lines.append(line)
            
            if self.first_frame_lines is None and new_lines:  # Jeśli jest to pierwsza klatka
                # Znajdź najdłuższą linię w funkcji anonimowej zwracajacej dlugosci wektorow 
                longest_line = max(new_lines, key=lambda line: np.sqrt((line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2))
                self.first_frame_lines = [longest_line]  # Zapisz najdłuższą linię
            lines = new_lines

        return lines


    # Rysowanie linii na klatce
    def drawLines(self, frame, lines):
        line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) # czarny obraz do narysowania linii odniesienia 
        if self.first_frame_lines is not None:  # Jeśli mamy linie z pierwszej klatki
            for line in self.first_frame_lines:  # Rysujemy na bialo 
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), [255, 255, 255], 2)
        # Dodanie obrazow do siebie i wagi
        frame_with_lines = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
        return frame_with_lines

# --------------------------------------------RZUTKI---------------------------------------------------
    # Procesowanie pojedynczej klatki i szukanie punktów na lotkach
    def processFrame(self, frame):

        blured = cv2.GaussianBlur(frame,(11,11),0)
        # Zamiana obrazu na HSV 
        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

        # Definiowanie zakresu koloru niebieskiego
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Tworzenie maski dla koloru niebieskiego
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("mask",mask)
        dart_points = []
        # wykluczanie zbyt małych konturów 
        for contour in contours:
            if cv2.contourArea(contour) < 80 :
                continue
            # Zwraca z konturu punkty x,y 
            (x, y, w, h) = cv2.boundingRect(contour)

            # zmiana orientacji z lewego wierzcholka na srodek wyznaczonego konturu
            dart_point = (x+w//2, y+h//2)
            dart_points.append(dart_point)
            
            # Obliczanie kąta i odległości dla każdego punktu rzutki
            angle, distance = self.compute_angle_and_distance(dart_point)
            self.CalculatePoints(angle,distance)
            self.DisplayPoints()
            # Sprawdzanie czy przekroczono 3 w liscie jezeli tak to usuwamy bo i tak sie powtarzaja a na koniec otrzymujemy dobry wynik
            if len(self.dart_points_info) >= 3:
                self.dart_points_info.pop(0)  
            
            self.dart_points_info.append((dart_point, angle, distance))
            
        return dart_points
    

    
    def drawVectorsAndAngles(self, frame):
        if self.first_frame_lines is not None and self.found_circle is not None:
            for info in self.dart_points_info:
                dart_point, angle, distance = info
                (x, y, r) = self.found_circle # paramtery tarczy 
                cv2.line(frame, (x-self.comp[0], y+self.comp[1]), dart_point, (255, 0, 0), 2)  # rysowanie wektora
                cv2.putText(frame, f"{angle:.2f}", (dart_point[0]+10, dart_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Wypisanie kąta obok rzutki

        return frame


    # Obliczanie kąta i odległości punktu rzutki względem linii bazowej
    def compute_angle_and_distance(self, dart_point):
        if self.first_frame_lines is not None and self.found_circle is not None:
            (x, y, r) = self.found_circle
            (dart_x, dart_y) = dart_point # wspolrzedne srodka konturu rzutki 
            ref_line = self.first_frame_lines[0][0]
            
            # Tworzenie wektora dla linii bazowej
            ref_vector = np.array([ref_line[2] - ref_line[0], ref_line[3] - ref_line[1]])
            
            # Tworzenie wektora dla rzutki. Odleglosc od srodka i miejsce gdzie trafiła lotka
            dart_vector = np.array([dart_x - (x-self.comp[0]), dart_y - (y+self.comp[1])])
            
            # Obliczanie kąta między dwoma wektorami
            angle = np.arctan2(dart_vector[1], dart_vector[0]) - np.arctan2(ref_vector[1], ref_vector[0])
            
            # Zamiana kąta z radianów na stopnie 
            angle = np.degrees(angle)
            # Obsługujemy tylko kąty -180 do 180 stopni 
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360
            
            # Obliczanie odległości od środka tarczy
            distance = np.sqrt((dart_x - x)**2 + (dart_y - y)**2)
            
            return angle, distance
        


    def CalculatePoints(self, angle,distance):
        
        # uzycie metody regulujacej warunki do punktacji
        points = self.CountPoints(angle, distance)

        if len(self.sepp_points) >= 3:
                self.sepp_points.pop(0)  
        self.sepp_points.append(points)
        # Zabezpieczenie przed liczeniem punktów w pętli i bardzo duzym wynikiem
        if len(self.sepp_points) == 3:
            self.total_points = self.sepp_points[0] + self.sepp_points[1] + self.sepp_points[2]
        


    def DisplayPoints(self):
        print(f"Total points: {self.total_points}")
        print("Points for each throw:")
        for index, points in enumerate(self.sepp_points, start=1):
            print(f"Throw {index}: {points} points")




    def CountPoints(self,angle,distance):
        points = 0
        x,y,r = self.found_circle

        if distance <= r*0.03:
            points = 50
        if distance <=r*0.08:
            points = 25
        if (distance < r*0.42 and distance > r*0.08) or (distance > r*0.47 and distance < r*0.7):
            if angle >= 0 and angle <= 18:
                points = 17 
            if angle >18 and angle <= 36:
                points = 3 
            if angle > 36 and angle <= 54:
                points = 19 
            if angle > 54 and angle <= 72:
                points = 7 
            if angle > 72 and angle <= 90:
                points = 16 
            if angle > 90 and angle <= 108:
                points = 8 
            if angle > 108 and angle <= 126:
                points = 11 
            if angle > 126 and angle <= 144:
                points = 14 
            if angle > 144 and angle <= 162:
                points = 9 
            if angle > 162 and angle <= 180:
                points = 12 
            if angle < 0 and angle >= -18:
                points = 2 
            if angle < -18 and angle >= -36:
                points = 15 
            if angle < -36 and angle >= -54:
                points = 10 
            if angle < -54 and angle >= -72:
                points = 6 
            if angle < -72 and angle >= -90:
                points = 13 
            if angle < -90 and angle >= -108:
                points = 4 
            if angle < -108 and angle >= -126:
                points = 18 
            if angle < -126  and angle >= -144:
                points = 1 
            if angle < -144 and angle >= -162:
                points = 20 
            if angle < -162 and angle >= -180:
                points = 5
        if distance >= r*0.42 and distance <= r*0.47:
            if angle >= 0 and angle <= 18:
                points = 17*3
            if angle >18 and angle <= 36:
                points = 3*3 
            if angle > 36 and angle <= 54:
                points = 19*3
            if angle > 54 and angle <= 72:
                points = 7*3 
            if angle > 72 and angle <= 90:
                points = 16*3
            if angle > 90 and angle <= 108:
                points = 8*3 
            if angle > 108 and angle <= 126:
                points = 11*3
            if angle > 126 and angle <= 144:
                points = 14*3
            if angle > 144 and angle <= 162:
                points = 9*3
            if angle > 162 and angle <= 180:
                points = 12*3
            if angle < 0 and angle >= -18:
                points = 2*3
            if angle < -18 and angle >= -36:
                points = 15*3
            if angle < -36 and angle >= -54:
                points = 10*3
            if angle < -54 and angle >= -72:
                points = 6*3 
            if angle < -72 and angle >= -90:
                points = 13*3 
            if angle < -90 and angle >= -108:
                points = 4*3
            if angle < -108 and angle >= -126:
                points = 18*3 
            if angle < -126  and angle >= -144:
                points = 1*3 
            if angle < -144 and angle >= -162:
                points = 20 *3
            if angle < -162 and angle >= -180:
                points = 5 *3
        if distance >= r*0.7 and distance <= r*0.75:
            if angle >= 0 and angle <= 18:
                points = 17*2
            if angle >18 and angle <= 36:
                points = 3*3
            if angle > 36 and angle <= 54:
                points = 19*2
            if angle > 54 and angle <= 72:
                points = 7*2 
            if angle > 72 and angle <= 90:
                points = 16*2
            if angle > 90 and angle <= 108:
                points = 8*2 
            if angle > 108 and angle <= 126:
                points = 11*2
            if angle > 126 and angle <= 144:
                points = 14*2
            if angle > 144 and angle <= 162:
                points = 9*2
            if angle > 162 and angle <= 180:
                points = 12*2
            if angle < 0 and angle >= -18:
                points = 2*2
            if angle < -18 and angle >= -36:
                points = 15*2
            if angle < -36 and angle >= -54:
                points = 10*2
            if angle < -54 and angle >= -72:
                points = 6*2 
            if angle < -72 and angle >= -90:
                points = 13*2 
            if angle < -90 and angle >= -108:
                points = 4*2
            if angle < -108 and angle >= -126:
                points = 18*2 
            if angle < -126  and angle >= -144:
                points = 1*2 
            if angle < -144 and angle >= -162:
                points = 20 *2
            if angle < -162 and angle >= -180:
                points = 5 *2
        return points
    
# --------------------------------------------Łączenie---------------------------------------------------
    # Procesowanie całego wideo
    def processVideo(self):
        video = cv2.VideoCapture(self.video_path)

        while(video.isOpened()):
            ret, frame = video.read()
            if ret:
                lines = self.findLines(frame) 
                
                self.findCircles(frame)
                dart_points = self.processFrame(frame)

                for point in dart_points:
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)
                cv2.imshow('Detected Darts', frame)

                frame_with_circles = self.drawCircles(frame)
                frame_with_lines = self.drawLines(frame_with_circles, lines)
                frame_with_circles = self.drawInnerCircles(frame_with_lines,self.comp)
            
                for info in self.dart_points_info:
                    print(f"Dart point: {info[0]}, Angle: {info[1]}, Distance: {info[2]}")
                cv2.imshow('Detected Circle and Lines', frame_with_circles)

                frame_with_circles = self.drawInnerCircles(frame_with_lines, self.comp)
                frame_with_vectors_and_angles = self.drawVectorsAndAngles(frame_with_circles)

                cv2.imshow('Detected Cir', frame_with_vectors_and_angles)
                if cv2.waitKey(1)  == ord('q'):
                    break
            else:
                break

        video.release()
        cv2.destroyAllWindows()

# Użycie klasy
detector = Detector('/Users/bartosz/Desktop/openCV_project/STUDYsession/trafia.MOV')
detector.processVideo()





