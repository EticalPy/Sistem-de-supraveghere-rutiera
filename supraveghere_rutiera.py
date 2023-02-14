import urllib.request
import uuid
import datetime
import torch
import cv2
import numpy as np
import easyocr
import os
import json
import urllib.request

 # Definim variabilele globale

EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2
lista_cautate = ['BCT5682', '3363FNJ', '7161GVN', '82RSXF']
counter = 0
imgBackground = cv2.imread('Resources/background12.png')
directory = r'/media/bogdan/SSD2/proiecte_2023_ubuntu/Supraveghere_rutiera/Capturate'
ip = "198.199.115.136"
GeoLocalizare = 'http://ip-api.com/json/'

 ### Functie de detectare a placutelor de inmatriculare

def detector(frame, model):

    frame = [frame]
    print(f"[INFO] Detectez numarul de inmatriculare...........")
    results = model(frame)

    etichete, coordonate = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return etichete, coordonate

 # Functia recunoasterii placutei de inmatriculare cu easyocr



def identificare_bbox(results, frame, clases):
    etichete, cord = results
    n = len(etichete)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    counter = 0
    print(f"[INFO] Total {n} detectate")
    print(f"[INFO] Analizez fiecare detectie in parte. . . ")

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            print(f"[INFO] Extrag coordonate detectie.....")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            coord = [x1, y1, x2, y2]

            plate_num = detectam_placuta_de_inmatriculare_cu_easyocr(img=frame, coords=coord, reader=EASY_OCR, region_threshold=OCR_TH)

            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1-40), (x2+40, y1), (0,255,0), -1)
            cv2.putText(frame, f"{plate_num}", (x1, y1-10),cv2.LINE_AA, 0.8, (0, 0, 255), 2)
            result_final = f"{plate_num}".replace(" ", "")
            if result_final in lista_cautate:
                print(f"[INFO] Am detectat autoturismul urmarit.....")
                counter += 1
                req = urllib.request.Request(GeoLocalizare + ip)
                response = urllib.request.urlopen(req).read()
                json_response = json.loads(response.decode('utf-8'))
                imgBackground2 = imgBackground.copy()
                imgBackground2[162:162 + 480, 55:55 + 640] = frame
                cv2.putText(imgBackground2, result_final, (770, 350), cv2.LINE_AA, 3, (0, 255, 0), 8)
                cv2.putText(imgBackground2,
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            (800, 570), cv2.LINE_AA, 1, (0, 255, 0), 2)
                LAT = json_response['lat']
                LONG = json_response['lon']
                ORAS = json_response['city']
                cv2.putText(imgBackground2, f"{LAT}", (1000,700), cv2.LINE_AA, 0.8, (0, 0, 255), 2)
                cv2.putText(imgBackground2, f"{LONG}", (650, 700), cv2.LINE_AA, 0.8, (0, 0, 255), 2)
                cv2.putText(imgBackground2, f"{ORAS}", (270, 700), cv2.LINE_AA, 0.8, (0, 0, 255), 2)
                #cv2.imshow("imagine", imgBackground2)
                if counter >= 1:
                    print(f"[INFO] Salvez fotografiile in dosarul Capturate")
                    img_name = '{}.jpg'.format(uuid.uuid1())
                    cv2.imwrite(os.path.join(directory, img_name), imgBackground2)

    return frame

def detectam_placuta_de_inmatriculare_cu_easyocr(img, coords, reader, region_threshold):
    xmin, ymin,xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    ocr_result = reader.readtext(nplate)

    text = filtru_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)
    if len(text) == 1:
        text = text[0].upper()

    return text

 # Definim functia care elimina rezultatele care nu ne intereseaza

def filtru_text(region, ocr_result, region_threshold):
    aria_dreptunghiului = region.shape[0] * region.shape[1]
    plate = []
    print(ocr_result)
    for result in ocr_result:
        lungimea = np.sum(np.subtract(result[0][1], result[0][0]))
        latimea = np.sum(np.subtract(result[0][2], result[0][1]))

        if lungimea * latimea / aria_dreptunghiului > region_threshold:
            plate.append(result[1])

    return plate

 # Definim functia principala

def main(vid_path=None, vid_out=None):
    print(f"[INFO] Incarc modelul invatat....")
    model = model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/media/bogdan/SSD2/proiecte_2023_ubuntu/Supraveghere_rutiera/model1/best.pt',
                                   force_reload=True)

    classes = model.names

    if vid_path != None:

        cap = cv2.VideoCapture(vid_path)
        cap.set(3, 640)
        cap.set(4, 480)

        if vid_out:   #### inregistram video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out,codec, fps, (width, height))

        frame_no = 1
        cv2.namedWindow('vid_out', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()

            if ret and frame_no % 1 == 0:
                print(f"[INFO] Analizez imaginea {frame_no}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = identificare_bbox(results, frame, clases=classes)

                cv2.imshow("vid_out", frame)


                if vid_out:
                    print(f"[INFO] Salvez inregistrarea")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        print(f"[INFO] Oprim inregistrarea ")
        out.release()

        cv2.destroyAllWindows()

main(vid_path=1, vid_out="supraveghere_rutiera.mp4")

