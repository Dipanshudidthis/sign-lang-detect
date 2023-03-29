import speech_recognition as sr
import cv2

ok=0
while(1):
    ok+=1
    r = sr.Recognizer()
    print("Converting Audio To Text ..... ")
    with sr.Microphone() as m:
        audio = r.listen(m)
        res = r.recognize_google(audio, language='end-in')
        print(res)

        if res!="":
            vid = cv2.VideoCapture("vids/{}.mp4".format(res))

            while (vid.isOpened()):
                ute, pic_fra = vid.read()
                if ute == True:
                    cv2.imshow('Frame', pic_fra)
                    if cv2.waitKey(25) & 0xFF == ord('u'):
                        break
                else:
                    break
            vid.release()
            cv2.destroyAllWindows()
        if(res=="end task"):
            break
