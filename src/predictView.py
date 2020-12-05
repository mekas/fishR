import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf
import os

class PredictView:

    def __init__(self, predictController):
        self.pc = predictController
        self.builder = Gtk.Builder()
        self.builder.add_from_file("glade/predict.glade")
        
        self.wd_predict = self.builder.get_object("wd_predict")
        self.wd_predict.set_title("FishR")
        self.wd_predict.connect("destroy", Gtk.main_quit)
        self.wd_predict.show_all()

        btn_select_img = self.builder.get_object("btn_select_img")
        btn_select_img.connect("clicked", self.on_btn_select_img)

        btn_select_model = self.builder.get_object("btn_select_model")
        btn_select_model.connect("clicked", self.on_btn_select_model)

        btn_predict = self.builder.get_object("btn_predict")
        btn_predict.connect("clicked", self.on_btn_predict)

    # select image
    # example : image.png
    def on_btn_select_img(self, widget):
        path_img = ''
        filename = ''

        dialog = Gtk.FileChooserDialog(
            title="Please choose a file", parent=None, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            print("Open clicked")
            path_img = dialog.get_filename()
            filename = os.path.basename(path_img)
            self.set_img_pred(path_img)
            self.set_lb_filename(filename)
            
            # set path_img
            self.pc.path_img(path_img)

        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")

        dialog.destroy()
        
    
    def set_img_pred(self, fullpath):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(fullpath, 224, 224)
        img_pred = self.builder.get_object("img_pred")
        img_pred.set_from_pixbuf(pixbuf)

    def set_lb_filename(self, message):
        lb_filename = self.builder.get_object("lb_filename")
        markup_text = "<span>"+ message + "</span>"
        lb_filename.set_markup(markup_text)
    
    # select model
    # example : model.h5
    def on_btn_select_model(self, widget):
        dialog = Gtk.FileChooserDialog(
            title="Please choose a file", parent=None, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            print("Open clicked")
            path_model = dialog.get_filename()
            filename = os.path.basename(path_model)
            self.set_btn_select_model(filename)

            # set path_model
            self.pc.path_model(path_model)

        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")

        dialog.destroy()

    def set_btn_select_model(self, label):
        btn_select_model = self.builder.get_object("btn_select_model")
        btn_select_model.set_label(label)

    # Predict
    def on_btn_predict(self, button):
        path_img = 'img.png'
        path_model = 'model.h5'
        result = 'test'

        print("path image: " + path_img)
        print("path model: " + path_model)
        print("result: " + result)
