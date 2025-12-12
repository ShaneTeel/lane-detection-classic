from typing import Union
from numpy.typing import NDArray

from lane_detection.studio.source import Reader
from lane_detection.studio.write import Writer
from lane_detection.studio.illustrate import Illustrator
from lane_detection.studio.render import Render
from lane_detection.studio.control import Controller
from lane_detection.studio.custodian import Custodian

class StudioManager():
    
    def __init__(self, source:Union[str, int], stroke_color:tuple = (0, 0, 255), fill_color:tuple = (0, 255, 0)):

        self.source = Reader(source)
        self.render = Render()
        self.write = Writer(self.source)
        self.draw = Illustrator(stroke_color=stroke_color, fill_color=fill_color)
        self.playback = Controller(self.source)
        self.clean = Custodian(self.source, self.write)
        self.exit = False

    def gen_view(self, frame_lst:list, frame_names:list=None, lines:list=None, view_style:str="original", stroke:bool=False, fill:bool=True):
        if view_style == "original":
            frame = self.draw._draw_banner_text(frame_lst[0], frame_names[0])
            return frame
        
        if view_style == "masked":
            frame = self.draw._draw_banner_text(frame_lst[1], frame_names[0])
            return frame
        
        if view_style == "diptych":
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render._render_diptych(frames)

        final = self.draw.gen_composite(frame_lst[0], lines, stroke, fill)
        del frame_lst[0]
        if view_style == "composite":
            return final
        if view_style == "inset":
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_inset(final, frames)
        if view_style == "mosaic":
            frame_lst.append(final)
            frames = [self.draw._draw_banner_text(frame, name) for frame, name in zip(frame_lst, frame_names)]
            return self.render.render_mosaic(frames, self.source.width)

    def return_frame(self):
        if self.source.source_type == 'image':
            return True, self.source.image
        
        if self.source.cap is None:
            return False, None

        ret, frame = self.source.cap.read()
        
        if ret:
            return True, frame
        else:
            return False, None

    def get_frame_names(self, view_style:str=None, method:str="final"):
        view_style_names = {
            "final": {
                "inset": ["HSL Thresh Mask", "Feature Map", "ROI Masked"],
                "mosaic": ["HSL Thresh Mask", "Feature Map", "ROI Masked", "Composite"],
                "composite": ["Composite"]
            },
            "preview": {
                "original": ["Original"],
                "masked": ["ROI Masked"],
                "diptych": ["Original", "ROI Masked"]
            }
        }

        if view_style is None:
            return "Composite"
        
        try:
            names = view_style_names[method][view_style]
            return names
        except Exception as e:
            raise KeyError(f"ERROR: Invalid argument ({e}) passed to 'view_style'. Must be one of {[key for key in view_style_names[method].keys()]}")
        
    def get_metadata(self):
        if self.source.fps is None:
            return 1, self.source.height, self.source.width
        return self.source.fps, self.source.height, self.source.width
    
    def create_writer(self, file_out_name:str, fourcc:str):
        self.write._initialize_writer(file_out_name, fourcc)

    def write_frames(self, frame:NDArray):
        if self.write.writer:
            self.write.save_object(frame)
        else:
            raise RuntimeError("ERROR: Never created writer object")
        
    def print_menu(self):
        self.playback.print_playback_menu()
    
    def source_type(self):
        return self.source.source_type
    
    def control_playback(self):
        return self.playback.playback_controls()
    
    def get_name(self):
        return self.source.name
    
    def writer_check(self):
        return True if self.write.writer is not None else False