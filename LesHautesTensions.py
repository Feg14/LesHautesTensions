from turtle import delay
import serial
import pyfiglet
import time


Ser = serial.Serial("COM3",9600)







#----------------------------------
print(pyfiglet.figlet_format("LES HAUTES TENSIONS"))

def commands():
    r=input("LES_HAUTES_TENSION>/")

    if ("set" in r)==True:
        arr=r.split(":")
        count=arr[1]
        Ser.write(r.encode("utf-8"))
    
        while r<=count:        
             text=str(Ser.readline())[2:][:-5]
             print(text)
     
    elif ("monitor" in  r)==True:
        arr=r.split(":")
        T=60*int(arr[1])
        timer=time.time()+T
        Ser.write(b"read") # appel la fonction "lire" de l'Arduino
        while time.time()<timer:
             text=str(Ser.readline())[2:][:-5]
             print(str(round(T-(timer-time.time())))+":"+text)
        Ser.write(b"exit")

    elif ("read" in r)==True:
        Ser.write(b"read") # appel la fonction "lire" de l'Arduino
        delay(1000)
        text=str(Ser.readline())[2:][:-5]
        print(text)
        Ser.write(b"exit")

    
    commands()
       



        

commands()


