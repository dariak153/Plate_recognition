import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import elasticdeform

def main():
    images = os.listdir(os.path.join('found_characters','to_filter'))
    for image in images:
        img = cv.imread(os.path.join('found_characters','to_filter',image))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        avg_gray = np.mean(gray)
        ret, thresh1 = cv.threshold(gray, avg_gray, 255, cv.THRESH_BINARY)
        cv.imwrite(os.path.join('found_characters', image), thresh1)

def main2():
    images = os.listdir(os.path.join('found_characters'))
    images = [image for image in images if image.endswith('.png')]
    # for image in images:
    #     img = cv.imread(os.path.join('found_characters',image))
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     gray_letter = gray[60:426, 144:512]
    #     gray = cv.resize(gray_letter, (120, 120))
    #     cv.imwrite(os.path.join('found_characters', image), gray)

    images_jpg = os.listdir(os.path.join('found_characters'))
    images_jpg = [image for image in images_jpg if image.endswith('.jpg')]

    for image in images_jpg:
        img = cv.imread(os.path.join('found_characters',image))
        image_name = image.split('.')[0]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (120, 120))
        cv.imwrite(os.path.join('found_characters', f'{image_name}.png'), gray)


def filter_image(image):
    deformed_image_filtered = cv.medianBlur(image, 5)
    _, deformed_image_filtered_1 = cv.threshold(deformed_image_filtered, 50, 255, cv.THRESH_BINARY)
    deformed_image_filtered_1 = cv.erode(deformed_image_filtered_1, np.ones((3, 3), np.uint8), iterations=1)
    return deformed_image_filtered_1
def main3():
    input_dir = 'found_characters'
    output_dir = 'randomized_characters'

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(input_dir)
    images = [image for image in images if image.endswith('.png')]

    images = sorted(images)
    letters = {}
    for image in images:
        letter = image[0]
        if letter not in letters:
            letters[letter] = 1
        else:
            letters[letter] += 1

    print(letters)
    each_letter = {}
    for letter in letters:
        each_letter[letter] = np.int32(np.ceil(150/letters[letter]))

    print(each_letter)

    for filename in images:
        # Read the image using OpenCV
        image_path = os.path.join(input_dir, filename)
        image = cv.imread(image_path)

        letter = filename[0]

        # Apply random elastic transformations 5 times
        for i in range(each_letter[letter]):
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Swap black and white
            image_gray = cv.bitwise_not(image_gray)
            if filename.endswith('.jpg'):
                image_gray = cv.medianBlur(image_gray, 5)
            transformed_image = elasticdeform.deform_random_grid(image_gray, sigma=0.75, points=np.random.randint(5, 15), rotate=np.random.uniform(-5, 5), zoom=np.random.uniform(0.9, 1.1))

            # Create the output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_{i + 1}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            # Filter the image
            gray_filtered = filter_image(transformed_image)

            if filename.endswith('.jpg'):
                def sharpen_image(image):
                    # Define a sharpening kernel
                    kernel = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])

                    # Apply the sharpening kernel to the image
                    sharpened = cv.filter2D(image, -1, kernel)
                    return sharpened

                # gray_filtered = filter_image(gray_filtered)
                # filter using gaussian blur
                gray_filtered = cv.GaussianBlur(gray_filtered, (11, 11), 0)
                gray_filtered = filter_image(gray_filtered)

            # Save the transformed image using OpenCV
            gray_filtered = cv.bitwise_not(gray_filtered)
            cv.imwrite(output_path, gray_filtered)

            # # Optionally display the result (uncomment to view the images)
            # plt.figure(1)
            # plt.imshow(transformed_image, cmap='gray')
            # plt.show()


    print("Randomized images have been saved successfully.")

def main4():
    for filename in os.listdir('odswinia'):
        image = cv.imread(os.path.join('odswinia', filename))
        file_name, ext = os.path.splitext(filename)
        if len(file_name) == 1:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, (120, 120))
            cv.imwrite(os.path.join('found_characters', f'{file_name}_1.png'), gray)
        else:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, (120, 120))
            gray = cv.bitwise_not(gray)
            cv.imwrite(os.path.join('found_characters', f'{file_name[0]}_2.png'), gray)

if __name__ == '__main__':
    #main()
    #main2()
    #main4()
    main3()