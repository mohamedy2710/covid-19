import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf

class LungClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Image Classification System")
        
        self.root.attributes('-fullscreen', True)
        
        self.FRAME_WIDTH = 800
        self.FRAME_HEIGHT = 600
        
        exit_button = ttk.Button(
            root,
            text="×",
            command=self.root.quit,
            width=10
        )
        exit_button.place(x=10, y=10)

        title_label = ttk.Label(
            root, 
            text="Lung Image Classification System", 
            font=('Arial', 24, 'bold')
        )
        title_label.pack(pady=20)
        
        try:
            bg_path = r"C:\Users\Ahmed\Desktop\CoronaVirus-Prediction-using-CNN-master\lungs.jpg"
            if os.path.exists(bg_path):
                bg_image = Image.open(bg_path)
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                bg_image = bg_image.resize((screen_width, screen_height))
                self.bg_photo = ImageTk.PhotoImage(bg_image)
                
                bg_label = tk.Label(root, image=self.bg_photo)
                bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                bg_label.lower()
        except Exception as e:
            print(f"Background image not found: {e}")

        self.main_frame = ttk.Frame(
            root,
            width=self.FRAME_WIDTH,
            height=self.FRAME_HEIGHT
        )
        self.main_frame.propagate(False)
        self.main_frame.place(
            relx=0.5,
            rely=0.5,
            anchor='center'
        )

        style = ttk.Style()
        style.configure('Main.TFrame', borderwidth=2, relief='solid')
        self.main_frame.configure(style='Main.TFrame')

        self.image_label = ttk.Label(self.main_frame)
        self.image_label.place(relx=0.5, rely=0.4, anchor='center')

        buttons_frame = ttk.Frame(root)
        buttons_frame.place(relx=0.5, rely=0.8, anchor='center')

        self.select_button = ttk.Button(
            buttons_frame,
            text="Select Image",
            command=self.select_image,
            width=20
        )
        self.select_button.pack(side='left', padx=20)

        self.classify_button = ttk.Button(
            buttons_frame,
            text="Classify Image",
            command=self.classify_image,
            state='disabled',
            width=20
        )
        self.classify_button.pack(side='left', padx=20)

        self.result_label = ttk.Label(
            root,
            text="",
            font=('Arial', 16)
        )
        self.result_label.place(relx=0.5, rely=0.85, anchor='center')

        self.root.bind('<Escape>', lambda e: self.root.quit())

        # Load the model
        self.model = tf.keras.models.load_model(r"C:\Users\Ahmed\Desktop\CoronaVirus-Prediction-using-CNN-master\model.h5")

        # Update to match the new class names
        self.class_names = ["COVID", "Normal"]

        self.selected_image = None
        self.photo = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        if file_path:
            image = Image.open(file_path)
            image = image.resize((self.FRAME_WIDTH-4, self.FRAME_HEIGHT-4))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.selected_image = file_path
            self.classify_button['state'] = 'normal'
            self.result_label['text'] = ""

    def classify_image(self):
        image = Image.open(self.selected_image)
        image = image.resize((224, 224))
        image = np.array(image)

        # If the image is grayscale, convert it to RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Predict using the model
        prediction = self.model.predict(image)

        # Print the raw prediction values to debug
        print("Prediction probabilities:", prediction)

        # As we have only two classes, use argmax to find the predicted class
        predicted_class = np.argmax(prediction)

        result_text = f"Result: {self.class_names[predicted_class]}"
        
        self.result_label['text'] = result_text

def main():
    root = tk.Tk()
    app = LungClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
