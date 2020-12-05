import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from predictView import PredictView


class MainController:
    def __init__(self):
        self.predictView = PredictView(self)

        Gtk.main()
    
    def path_img(self, path_img):
        t_path_img = path_img
        print("path image : "+ t_path_img)

        return t_path_img

    def path_model(self, path_model):
        t_path_model = path_model
        print("path model : " + t_path_model)

        return t_path_model


main = MainController()