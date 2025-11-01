import cv2
import numpy as np

def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]        # Top-Left
    rect[2] = pts[np.argmax(s)]        # Bottom-Right
    rect[1] = pts[np.argmin(diff)]     # Top-Right
    rect[3] = pts[np.argmax(diff)]     # Bottom-Left
    return rect

def countScore(img):
    
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10 
    )

    # Find control squares
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 20000: # Image size
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                squares.append(c)

    # Select the 4 biggest
    squares = sorted(squares, key=cv2.contourArea, reverse=True)[:4]

    img_contours = original.copy()
    cv2.drawContours(img_contours, squares, -1, (0, 0, 255), 3)

    # 5. Calculate the point to cut image
    points = []
    i = 0
    for c in squares:
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        squareT = 30 # Change with the size of the squares
        if(i == 0):
            points.append([cx+squareT, cy+squareT])
            cv2.circle(img_contours, (cx+squareT, cy+squareT), 10, (0, 255, 0), -1)
        elif(i == 1):
            points.append([cx-squareT, cy+squareT])
            cv2.circle(img_contours, (cx-squareT, cy+squareT), 10, (0, 255, 0), -1)
        elif(i == 2):
            points.append([cx-squareT, cy-squareT])
            cv2.circle(img_contours, (cx+squareT, cy-squareT), 10, (0, 255, 0), -1)
        elif(i == 3):
            points.append([cx+squareT, cy-squareT])
            cv2.circle(img_contours, (cx-squareT, cy-squareT), 10, (0, 255, 0), -1)
        
        i = i + 1
        
    src_pts = order_points(points)

    width, height = 1000, 1400

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(original, M, (width, height))

    result_img = aligned.copy()
    
    gray_img = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Find grid lines
    img_threshold = cv2.adaptiveThreshold(
        img_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=25, 
        C=10           
    )

    bordas = cv2.Canny(img_blur, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(bordas, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    lines_h = []
    lines_v = []

    if lines is not None:
        for linha in lines:
            x1, y1, x2, y2 = linha[0]
            
            if abs(x2 - x1) < 5:
                lines_v.append(x1) 
            elif abs(y2 - y1) < 5:
                lines_h.append(y1)

    # Remove duplicated lines
    lines_h = sorted(list(set([y for y in lines_h if y > 0])))
    lines_v = sorted(list(set([x for x in lines_v if x > 0])))
    
    def filter_next_lines(coord, tol=30):
        if not coord:
            return []
        
        filtered = [coord[0]]
        for coord in coord[1:]:
            if abs(coord - filtered[-1]) > tol:
                filtered.append(coord)
        return filtered

    lines_h = filter_next_lines(lines_h)
    lines_v = filter_next_lines(lines_v)

    # Detect rectangles in grid
    rect_grid = []
    
    for i in range(len(lines_h) - 1):
        y1 = lines_h[i]
        y2 = lines_h[i+1]
        for j in range(len(lines_v) - 1):
            x1 = lines_v[j]
            x2 = lines_v[j+1]
            
            # x, y, w, h
            rect_grid.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Verify 'X'
    
    rect_with_x = 0
    
    img_to_detect_x = img_threshold 
    
    for x, y, w, h in rect_grid:
        
        cut_margin = 15 # Increase to ignore grid lines
        
        if w > 2 * cut_margin and h > 2 * cut_margin:
            roi_without_border = img_to_detect_x[y + cut_margin : y + h - cut_margin, 
                                                  x + cut_margin : x + w - cut_margin]
            inside_w, inside_h = roi_without_border.shape[1], roi_without_border.shape[0]
        else:
            roi_without_border = img_to_detect_x[y:y+h, x:x+w]
            inside_w, inside_h = w, h

        if inside_w <= 0 or inside_h <= 0:
            continue

        # find contours in area without borders
        
        has_x = False
        candidates_x = []
        area_cell = w * h

        int_contours, _ = cv2.findContours(roi_without_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for inside_c in int_contours:
            area_inside = cv2.contourArea(inside_c)
            
            # Filter X area
            if area_cell * 0.01 < area_inside < area_cell * 0.75:
                
                _, _, w_c, h_c = cv2.boundingRect(inside_c)
                area_bounding_box = w_c * h_c
                
                if area_bounding_box > 0:
                    solid_rate = area_inside / area_bounding_box
                    
                    # Filter Solid rate
                    if 0.1 < solid_rate < 0.7: 
                        candidates_x.append(inside_c)
        
        # Verify if the X is in the cell center and count
        if 1 <= len(candidates_x) <= 4:
            main_c = max(candidates_x, key=cv2.contourArea)
            M = cv2.moments(main_c)
            if M["m00"] != 0:
                cx_inside = int(M["m10"] / M["m00"])
                cy_inside = int(M["m01"] / M["m00"])
                
                center_roi_x = inside_w // 2
                center_roi_y = inside_h // 2
                
                limit_dist_x = inside_w * 0.5 
                limit_dist_y = inside_h * 0.5
                
                if abs(cx_inside - center_roi_x) < limit_dist_x and \
                   abs(cy_inside - center_roi_y) < limit_dist_y:
                    has_x = True

        if not has_x:
            # Another verification to check if there is some marking in the cell
            black_pixels = cv2.countNonZero(roi_without_border)
            roi_size = inside_w * inside_h

            if roi_size > 0:
                # Calculate % of cell that have something
                percent_fill = black_pixels / roi_size
                
                if 0.07 < percent_fill < 0.80:
                    has_x = True
        
        if has_x:
            rect_with_x += 1
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return rect_with_x