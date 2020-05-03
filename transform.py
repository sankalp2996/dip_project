import numpy as np
from math import ceil, sqrt, exp
from math import pi as PI


class Math:
    @staticmethod
    def normalize(array):
        return Math.scale(array, 1. / sum(array))

    @staticmethod
    def cumulative(array):
        result = [0] * len(array)
        result[0] = array[0]
        for i in range(1, len(array)):
            result[i] = result[i - 1] + array[i]
        return result

    @staticmethod
    def scale(array, m):
        return [m * array[i] for i in range(len(array))]


class Histogram:
    @staticmethod
    def hist(image):
        h = [0] * 256
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                h[image[y, x]] += 1
        return h

    @staticmethod
    def uniform(center, width):
        h = [0] * 256
        i = int(ceil(center - width * 0.5))
        j = int(ceil(center + width * 0.5))
        for k in range(i, j):
            h[k % 256] = 1
        return h

    @staticmethod
    def triangle(center, width):
        h = [0] * 256
        i = int(ceil(center - width * 0.5))
        j = int(ceil(center + width * 0.5))
        for k in range(i, center):
            h[k % 256] = k - i
        for k in range(center, j):
            h[k % 256] = (center - i) - (k - center)
        return h

    @staticmethod
    def gaussian(mu, sigma):
        return [(1 / sqrt(2 * PI * sigma ** 2)) * exp(-(k - mu) ** 2 / (2. * sigma ** 2)) for k in range(256)]

    @staticmethod
    def repeat(hist, repeatTimes, repeatDistance, repeatDirection):
        result = hist[:]
        if repeatDirection != "right":
            for k in range(1, 1 + repeatTimes):
                d = k * repeatDistance
                for i in range(256):
                    result[i] = max(result[i], hist[(i + d) % 256])
        if repeatDirection != "left":
            for k in range(1, 1 + repeatTimes):
                d = k * repeatDistance
                for i in range(256):
                    result[i] = max(result[i], hist[(i - d) % 256])
        return result


class Transform:
    @staticmethod
    def negative(matrix):
        return 255 - matrix

    @staticmethod
    def log(matrix, c):
        return (c * np.log(matrix + 1)).round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def power(matrix, c, p):
        return (c * np.power(matrix, p)).round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def equalize(image, hist):
        cdf = Math.cumulative(hist)
        cdf = Math.scale(cdf, 255. / cdf[-1])
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                image[i, j] = cdf[image[i, j]]
        return image

    @staticmethod
    def shape(img, hist, function, center, width, repeatTimes, repeatDistance,
              repeatDirection):
        # generate target histogram
        functions = {
            'uniform': Histogram.uniform,
            'triangle': Histogram.triangle,
            'gaussian': Histogram.gaussian
        }
        fun = functions[function]
        target_hist = fun(center, width)
        target_hist = Histogram.repeat(target_hist, repeatTimes, repeatDistance,
                                       repeatDirection)
        # match hist to target_hist
        return (Transform.match(img, hist, target_hist), target_hist)

    @staticmethod
    def match(img, hist, hist_match):
        # normalize histograms
        hist = Math.normalize(hist)
        hist_match = Math.normalize(hist_match)
        # cumulative histograms
        hist = Math.cumulative(hist)
        hist_match = Math.cumulative(hist_match)
        # generate a map
        output = [0] * 256
        for i in range(256):
            j = 256 - 1
            while True:
                output[i] = j
                j = j - 1
                if j < 0 or hist[i] > hist_match[j]:
                    break
        # map the intensities
        height, width = img.shape
        for i in range(height):
            for j in range(width):
                temp = img[i][j]
                temp2 = output[temp]
                img[i][j] = temp2

        return img
