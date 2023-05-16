import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk





def processing(directory):
    #directory="/Users/francois-etienne/Documents/Programmation/Python/Analyse-Bacterie/SE 24 avril/Pétris ozonées/"
    print("LES_HAUTES_TENSIONS>/ Le traitement des donées à commencé...")
    start_time=time.time()

    #directory="photos/"

    # get the list of all files and directories in the directory
    dir_contents = os.listdir(directory)

   
    
    
    for filename in os.listdir(directory):
      if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        figure, axis =plt.subplots(1, 3, figsize=(30, 15))
        img = cv2.imread(os.path.join(directory, filename))
       



        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)






        #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 5,3000,param1=500)


        ret,thresh = cv2.threshold(gray,127,255,0)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
        for cnt in contours:
          (x,y),radius = cv2.minEnclosingCircle(cnt)
          center = (int(x),int(y))
          radius = int(radius)
          if radius>1000:
            print("Cercle détecté!")
            radius=radius-250







            cv2.circle(img,center,radius,(0,255,43),15)
            axis[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            axis[0].set_title(str(filename))
            axis[0].axis("off")

            
            # Create a mask with the same dimensions as the original image
            mask = np.zeros_like(gray)

            # Set the pixels outside the circle to zero
            cv2.circle(mask, center, radius, 255, -1)

            # Apply the mask to the original image
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            #Parameter for cropping
            x_start = center[0] - radius
            y_start = center[1] - radius
            x_end = center[0] + radius
            y_end = center[1] + radius
            cropped_img = masked_img[y_start:y_end, x_start:x_end]
            
    #Brigthness calibration----------------------------------------
            #gamma = 1
            # Apply gamma correction to the image
            #corrected_image = np.power(cropped_img / 255.0, gamma)
            #corrected_image = np.uint8(corrected_image * 255)


            #img_yuv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2YUV)

            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            #img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
            # convert the YUV image back to RGB format
            #cropped_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    #--------------------------------------------------------------
            #plot the cropped image

            image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)


        
          

            # reshape the image to a 2D array of pixels and 3 color values (RGB)
            pixel_values = image.reshape((-1, 3))
            # convert to float
            pixel_values = np.float32(pixel_values)


            # define stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500,0.1)
            # number of clusters (K)
            k = 3
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 5,  cv2.KMEANS_RANDOM_CENTERS)
            # convert back to 8 bit values
            centers = np.uint8(centers)



            # flatten the labels array
            labels = labels.flatten()
            # convert all pixels to the color of the centroids
            segmented_image = centers[labels]
            # reshape back to the original image dimension
            segmented_image = segmented_image.reshape(image.shape)
   

            print("Calcul degré de confiance...")
            #silhouette_avg = silhouette_score(pixel_values, labels)

            #print("Silhouette Coefficient:", silhouette_avg)



            #Pie plot with cluster percentage------------------------------------
            unique, counts = np.unique(labels, return_counts=True)
            labels = centers[0],centers[1],centers[2]
            sizes = counts
           

            rgb = centers[0]
            color_code0 = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

            rgb = centers[1]
            color_code1 = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])    

            rgb = centers[2]
            color_code2 = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])                  

            axis[2].pie(sizes, labels=labels,textprops={'color': '#FF00FF','fontsize':'14'},autopct='%1.1f%%',colors=[color_code0,color_code1,color_code2])

            axis[1].imshow(segmented_image)
            #axis[1].set_title("Rayon en pixel: "+str(radius))
            axis[1].axis("off")

            end_time=time.time()
            temps=str(round(end_time-start_time,0))
            print("LES_HAUTES_TENSIONS>/ Le traitement des données est terminé!")
            plt.savefig(directory+"Output/"+filename)
            
            plt.show()
            
          
        







class App:
    def __init__(self, root):
        # setting title
        root.title("LES HAUTES TENSIONS")
        # setting window size
        width = 400
        height = 200
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        # create a label and an entry
        self.label = tk.Label(root, text="Chemins d'accès des images:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root,width=40)
        self.entry.pack(pady=10)

        # create a button that prints the entry result
        self.button = tk.Button(root, text="Démarrer le traitement", command=self.print_entry)
        self.button.pack(pady=10)

    def print_entry(self):
        # get the text from the entry and print it
        entry_text = self.entry.get()
        print(entry_text)
        processing(entry_text)



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()






