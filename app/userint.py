# importing kivy depedencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
#import kivy uix components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import numpy as np


# building user interface
class CamApp(App):

    def build(self):
        # main layout
        self.img1 = Image(size_hint=(1, .8))
        self.button = Button(text="start", size_hint=(1, .1))
        self.pos = Label(text="neutral", size_hint=(1, .1))

        # add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.pos)

        # setting up vied capture device
        self.capture = cv2.VideoCapture(1)

        return layout

  #run continiously to get webcafeed
  def update(self, *args):
      ret, frame = self.caputre.read()
      frame = [120:120+250, 200:200+250, :]

    buf = cv2.flip(frame, 0).tostring()
    img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='brg'))
    img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    self.img1.texture=img_texture

if __name__ == '__main__':
    CamApp().run()