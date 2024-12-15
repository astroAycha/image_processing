import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def crop_face(image_path, output_path, padding_ratio=0.3):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5, 
                                          minSize=(30, 30))

    no_face = []
    if len(faces) == 0:
        no_face.append(image_path)
        with open('faulty_face_detection.csv', 'a') as f:
            f.write(image_path+'\n')
        return no_face

    # Assume the first face detected is the face we want to crop
    x, y, w, h = faces[0]
    
    # Add padding around the face by expanding the bounding box
    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)

    # Calculate new bounding box coordinates
    x_new = max(x - padding_x, 0)
    y_new = max(y - padding_y, 0)
    w_new = min(x + w + padding_x, img.shape[1]) - x_new
    h_new = min(y + h + padding_y, img.shape[0]) - y_new
    
    # Crop the image using the coordinates of the first detected face
    cropped_img = img[y_new:y_new+h_new, x_new:x_new+w_new]

    # Convert the cropped image back to RGB (for saving with Pillow)
    cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    
    cropped_img_pil.save(output_path)


def add_watermark(base_image_path, 
                  watermark_image_path, 
                  output_image_path, 
                  position="bottom-right", 
                  opacity=0.8):
    # Open the base image and watermark image
    try:
        base_image = Image.open(base_image_path)
        watermark = Image.open(watermark_image_path)

        # Resize watermark to match the size of the base image
        base_width, base_height = base_image.size
        watermark = watermark.resize((base_width, base_height), Image.Resampling.LANCZOS)

        # Add opacity to watermark
        if watermark.mode != 'RGBA':
            watermark = watermark.convert("RGBA")
        
        # Create an alpha mask based on opacity
        watermark = ImageEnhance.Brightness(watermark).enhance(opacity)

        # Get watermark dimensions (now matching base image size)
        watermark_width, watermark_height = watermark.size

        # Determine the position to place the watermark
        if position == "center":
            position = (base_width // 2 - watermark_width // 2, base_height // 2 - watermark_height // 2)
        elif position == "bottom-right":
            position = (base_width - watermark_width - 10, base_height - watermark_height - 10)
        elif position == "top-left":
            position = (10, 10)
        else:
            raise ValueError("Invalid position specified. Use 'center', 'bottom-right', or 'top-left'.")

        # Paste the watermark onto the base image
        base_image.paste(watermark, position, watermark)

        # Save the output image
        base_image.save(output_image_path, "PNG")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        
