import cv2

class ImageProcessor:

    def __init__(self):
        self.contours = None
        self.image = None
        self.contour_center_x = 0
        self.main_contour = None
        self.direction = 0

    def process(self):
        if self.image is None:
            raise ValueError("Image not set")

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to Gray Scale
        self.contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Get contours

        if self.contours:
            self.main_contour = max(self.contours, key=cv2.contourArea)
            height, width = self.image.shape[:2]
            middle_x = width // 2  # Get X coordinate of the middle point
            middle_y = height // 2  # Get Y coordinate of the middle point

            prev_contour_center_x = self.contour_center_x
            contour_center = self.get_contour_center(self.main_contour)
            if contour_center:
                self.contour_center_x = contour_center[0]
                if abs(prev_contour_center_x - self.contour_center_x) > 5:
                    self.correct_main_contour(prev_contour_center_x)
            else:
                self.contour_center_x = 0

            self.direction = int((middle_x - self.contour_center_x) * self.get_contour_extent(self.main_contour))

            self.draw_contours(middle_x, middle_y)

    @staticmethod
    def get_contour_center(contour):
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        return x, y

    @staticmethod
    def get_contour_extent(contour):
        area = cv2.contourArea(contour)
        _, _, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area > 0:
            return float(area) / rect_area
        return 0

    @staticmethod
    def approx(a, b, error):
        return abs(a - b) < error

    def correct_main_contour(self, prev_cx):
        for contour in self.contours:
            contour_center = self.get_contour_center(contour)
            if contour_center:
                tmp_cx = contour_center[0]
                if self.approx(tmp_cx, prev_cx, 5):
                    self.main_contour = contour
                    new_center = self.get_contour_center(self.main_contour)
                    if new_center:
                        self.contour_center_x = new_center[0]
                        break

    def draw_contours(self, middle_x, middle_y):
        cv2.drawContours(self.image, [self.main_contour], -1, (0, 255, 0), 3)  # Draw Contour GREEN
        cv2.circle(self.image, (self.contour_center_x, middle_y), 7, (255, 255, 255), -1)  # Draw dX circle WHITE
        cv2.circle(self.image, (middle_x, middle_y), 3, (0, 0, 255), -1)  # Draw middle circle RED

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, str(middle_x - self.contour_center_x),
                    (self.contour_center_x + 20, middle_y), font, 1, (200, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(self.image, f"Weight:{self.get_contour_extent(self.main_contour):.3f}",
                    (self.contour_center_x + 20, middle_y + 35), font, 0.5, (200, 0, 200), 1, cv2.LINE_AA)