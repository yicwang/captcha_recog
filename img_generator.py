from captcha.image import ImageCaptcha
import random

RANDSETS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
SAMPLES = 1024

def get_random_text(length):
    retVal = ''
    for _ in range(length):
        retVal += RANDSETS[random.randint(0, len(RANDSETS)-1)]
    return retVal

if __name__ == '__main__':
    ic = ImageCaptcha()
    for _ in range(SAMPLES):
        text = get_random_text(4)
        # image = ic.generate(text, format='png')
	ic.write(text, "%s.png" % text)
