import os
import pickle
import argparse
import json
from tools.utils import detect_characters
import numpy as np
import cv2
import matplotlib.pyplot as plt

chars_to_detect = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'V', 'Y', 'Z',
                   'X', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_dir', type=str)
    args=parser.parse_args()
    images_dir = args.images_dir
    results_dir = args.results_dir


    with open('characters_recognizer_rf.pkl', 'rb') as f:
        model = pickle.load(f)

    output_json = {}
    for license_plate in os.listdir(images_dir):
        characters = detect_characters(os.path.join(images_dir, license_plate))

        print("Characters: ", len(characters))

        recognized_plate = ''
        for character in characters:
            #recognized_character = model.predict(character.flatten().reshape(1, -1))
            recognized_character = model.predict_proba(character.flatten().reshape(1, -1))
            print("Character: ", chars_to_detect[np.argmax(recognized_character)], "Probability: ", np.max(recognized_character))
            # plt.imshow(character, cmap='gray')
            # # plt.show()
            # while True:
            #      print(recognized_character)
            #      plt.imshow(character, cmap='gray')
            #      plt.show()
            #      cv2.imshow("letter", character)
            #      key = cv2.waitKey(0)
            #      if key == ord('q'):
            #         break
            if np.max(recognized_character) >= 0.21:
                recognized_plate += chars_to_detect[np.argmax(recognized_character)]

            if len(recognized_plate) == 8:
                break

        print(recognized_plate)

        for i in range(2):
            if recognized_plate[i] == '0':
                recognized_plate = recognized_plate[:i] + "O" + recognized_plate[i + 1:]
            if recognized_plate[i] == '1':
                recognized_plate = recognized_plate[:i] + "I" + recognized_plate[i + 1:]
            if recognized_plate[i] == '2':
                recognized_plate = recognized_plate[:i] + "Z" + recognized_plate[i + 1:]
            if recognized_plate[i] == '3':
                recognized_plate = recognized_plate[:i] + "E" + recognized_plate[i + 1:]
            if recognized_plate[i] == '4':
                recognized_plate = recognized_plate[:i] + "K" + recognized_plate[i + 1:]
            if recognized_plate[i] == '5':
                recognized_plate = recognized_plate[:i] + "S" + recognized_plate[i + 1:]
            if recognized_plate[i] == '6':
                recognized_plate = recognized_plate[:i] + "O" + recognized_plate[i + 1:]
            if recognized_plate[i] == '7':
                recognized_plate = recognized_plate[:i] + "T" + recognized_plate[i + 1:]
            if recognized_plate[i] == '8':
                recognized_plate = recognized_plate[:i] + "B" + recognized_plate[i + 1:]
            if recognized_plate[i] == '9':
                recognized_plate = recognized_plate[:i] + "R" + recognized_plate[i + 1:]

        output_json[license_plate] = recognized_plate

    with open(results_dir, 'w') as file:
        json.dump(output_json, file)