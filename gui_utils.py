import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

def select_file(start_dir=None):
    result= []

    def run_dialog(_None):
        dialog = Gtk.FileChooserDialog(title="Select an image file",
            parent=None,
            action=Gtk.FileChooserAction.OPEN,)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                     Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        if start_dir is not None:
            dialog.set_current_folder(start_dir)

        dialog.set_keep_above(True)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            result.append(dialog.get_filename())
        else:
            result.append(None)

        dialog.destroy()
        Gtk.main_quit()

    Gdk.threads_add_idle(GLib.PRIORITY_DEFAULT, run_dialog, None)
    Gtk.main()
    return result[0]
