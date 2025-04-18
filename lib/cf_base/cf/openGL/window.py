import time
import tkinter as tk
from tkinter import filedialog
import logging
import numpy as np

from cf.openGL import *
from cf.openGL.framebuffer import Framebuffer
from cf.openGL.operations.display import ImageDisplayOP
from cf.openGL.operations.text import TextOP
from cf.images.image_io import save_image
from cf.tools.string_tools import print_same_line
from cf.mathematics.matrix import isotropic_scaling_matrix_2D, translation_matrix_2D

# An interactive OpenGL window, handling mouse and keyboard inputs.
class OpenGLWindow:

    def __init__(self, display_res=(1,1), title="OpenGL Window", logging_level=logging.INFO, max_fps = None):
        
        assert OGL_FRAMEWORK == OpenGLFramework.glfw, "Can only use GLFW to create a window"

        self.window_handle = glfw.create_window(display_res[0], display_res[1], title, None, None)
        assert self.window_handle, "Unable to create GLFW window"
        glfw.make_context_current(self.window_handle)    

        self.fbo = Framebuffer()

        glfw.set_key_callback(self.window_handle, self.key_callback)
        glfw.set_mouse_button_callback(self.window_handle, self.mouse_click_callback)
        glfw.set_cursor_pos_callback(self.window_handle, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window_handle, self.mouse_scroll_callback)

        logging.basicConfig(level=logging_level, format='%(levelname)s [%(funcName)s]: %(message)s')

        self.display_res = display_res
        self.max_fps = max_fps
        self.calc_fps = None

        self.mods = 0

        self.mouse_is_clicked = False
        self.mouse_is_moved = False

        self.left_mouse_down = False
        self.right_mouse_down = False
        self.middle_mouse_down = False
        self.click_position = None

        self.display_count = 1
        self.display_idx = 1
        self.display_stats = True
        self.stats_frequency = 0.25

        self.is_recording_frame = False 
        self.is_recording_display = False
        self.frame_recording_buffer = []

        self.frame_count = 0
        self.lazy_update = False
        self.need_update = True
        
        try:
            self.init_shaders()
        except:
            logging.debug("Could not initialize shaders in OpenGLWindow. Manually call init_shaders() in derived class.")

        self.fbo.bind()

        print("----- Press H for help window. -----")

        # Default keys: # Add more key bindings as needed using the self.register_key method..
        # update this dict if keys are within some range...
        self.key_bindings = {
            "0 & 1-9" : "Set Display 10, 1-9"
            
        }

        # Update help dictionary
        self.default_key_press(None)
        self.key_press(None)


    #----------------------------------

    # initialize all shaders
    def init_shaders(self):
        self.display_op = ImageDisplayOP(self.display_res)
        self.performance_text_op = TextOP((250, 37), font_size=17)

    #----------------------------------

    # starts the main display loop
    def run(self):
        logging.info("Entering main loop.")
        last_performance_update = time.perf_counter() 
        performance_tex = None
        self.render_tex = None
        self.fps_count = 0
        smoothing_factor = 0.9
        warm_up_time = 10
        while not glfw.window_should_close(self.window_handle):
            if self.need_update:
                self.fbo.bind()
                time_before = time.perf_counter() 
                textures = self.render() # the main render function
                if self.display_stats or self.is_recording_display:
                    glFinish()
                    time_after = time.perf_counter()
                    if (time_after - last_performance_update) > self.stats_frequency and self.fps_count > warm_up_time:
                        time_elapsed = (time_after - time_before) * 1000
                        stats_text = (  f"[{self.display_idx}/{self.display_count}]    "
                                        f"{time_elapsed:0.1f} ms ({self.max_fps if self.max_fps != None else self.calc_fps} fps) ")
                        
                        if self.is_recording_display: 
                            stats_text += f"[{self.frame_count}]" 
                        stats_color = [0, 0, 0] if self.is_recording_display else [1, 1, 1]
                        background_color = [0.8, 0, 0, 0.7] if self.is_recording_display else [0, 0, 0, 0.7]
                        performance_tex = self.performance_text_op.render(stats_text, position=[15, 12], color=stats_color, background_color=background_color)
                        last_performance_update = time_after
                else:
                    performance_tex = None
                self.fbo.unbind()

                # ----------------------------------------------------

                if textures is not None:
                    if not isinstance(textures, tuple):
                        textures = (textures,)
                    self.display_count = len(textures)
                    self.display_idx = min(self.display_idx, self.display_count)
                    self.render_tex = textures[self.display_idx-1]
                    if self.is_recording_frame:
                        self.frame_recording_buffer.append(self.render_tex.download_image())
                        self.frame_count += 1

                # ----------------------------------------------------
                
                if self.fps_count > warm_up_time:
                    self.need_update = False

            self.display_op.render(self.render_tex, overlay_tex=performance_tex, to_screen=True)
            glfw.swap_buffers(self.window_handle)
            glfw.poll_events()

            self.post_render()    
            
            # compute_fps
            # ----------------------------------------------------
            if self.need_update:
                time_end = time.perf_counter()
                render_time = time_end - time_before
                if self.max_fps != None:
                    if render_time < (1 / self.max_fps):
                        time.sleep((1 / self.max_fps) - render_time)
                else:
                    self.calc_fps = 1/render_time
                    time.sleep(1e-100)
                
                if not self.fps_count == 0:
                    # Exponential moving average
                    self.calc_fps = (self.calc_fps * smoothing_factor) + (self.prev_fps * (1 - smoothing_factor))
                                
                self.calc_fps = round(self.calc_fps)
                self.prev_fps = self.calc_fps

                self.fps_count += 1
            
            # ----------------------------------------------------
            
        glfw.terminate()

    #----------------------------------
    @property
    def need_update(self):
        return self._need_update
    
    @need_update.setter
    # set flag to compute the render pass
    def need_update(self, value):
        if self.lazy_update:
            # compute frame only when a click and dragging motion is performed..pure mouse movement doesnot affect anything
            if self.mouse_is_moved:
                if self.mouse_is_clicked:
                    self._need_update = value
                self.mouse_is_moved = not self.mouse_is_moved
                return
            
            self._need_update = value
        else:
            self._need_update = True

    #----------------------------------

    # overwrite this function with your custom render code
    def render(self):
        pass
    
    #----------------------------------
    # overwrite this function with your custom post render code
    def post_render(self):
        pass
    
    #----------------------------------

    def save_frames(self):
        root = tk.Tk()
        root.withdraw()
        recording_directory = filedialog.askdirectory()
        if recording_directory:
            for idx, frame in enumerate(self.frame_recording_buffer):
                print_same_line(f"Saving frame {idx+1}/{len(self.frame_recording_buffer)} to disk.")
                save_image(frame, os.path.join(recording_directory, f"recording_{idx:05}.png"))
            print("")
        self.frame_count = 0
        self.frame_recording_buffer = []

    #----------------------------------

    def register_key(self, register_key, match_key=None, description=""):
        if match_key is None:
            self.key_bindings.update({
                register_key: f"{description}"
                })
            return False
        if isinstance(register_key, tuple):
            return register_key[1] == match_key and self.mods == register_key[0]
        else:
            # When no modifiers are present, strictly activate the registered key.
            # This (self.mods == 0) prevents collision with the above condition for keys with modifiers.
            return register_key == match_key and self.mods == 0

            
    #----------------------------------

    # handle key inputs
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.need_update = True
            self.default_key_press(key)
            self.key_press(key)
            
        if action == glfw.RELEASE:

            # register shift and ctrl release
            if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                self.mods = 0
            if key == glfw.KEY_LEFT_CONTROL or key == glfw.KEY_RIGHT_CONTROL:
                self.mods = 0

            self.key_release(key)

    #----------------------------------

    # overwrite these functions with custom key events
    def key_press(self, key):
        pass
        
    def default_key_press(self, key):
        def glfw_key(key):
            return key_mappings.get(key, f"{key}")
        
        if self.register_key(glfw.KEY_ESCAPE, key, "Exit main loop."):
            logging.info("Exiting main loop.")
            glfw.set_window_should_close(self.window_handle, True)

        if self.register_key(glfw.KEY_H, key, "Show Help Window."):
            print()
            print("------- Help Window ----------")
            print()
            for k, description in self.key_bindings.items():
                if isinstance(k, tuple):
                    readable_key = "+".join([glfw_key(key) for key in k])
                else:
                    readable_key = glfw_key(k)

                # Beautify layout # 30 coz that's the 2* (maximum string length available)
                if len(readable_key) < 30:
                    readable_key += ' ' * (30 - len(readable_key)) 
                tabs = '\t' * 2

                formatted_string = f"{readable_key}{tabs}{description}"
                print(formatted_string)
            print()
            print("------------------------------")
            print()

        # display stats
        if self.register_key(glfw.KEY_P, key, "Toggle display stats."):
            self.display_stats = not self.display_stats

        # Lazy update display
        if self.register_key(glfw.KEY_T, key, "Toggle lazy update."):
            self.lazy_update = not self.lazy_update
            self.need_update = not self.need_update
            self.fps_count = 0
            logging.info(f"Lazy Update : {self.lazy_update}")

        # reload shaders
        if self.register_key(glfw.KEY_R, key, "Reload shaders."):
            logging.info("Reloading shaders.")
            try:
                self.init_shaders()
            except RuntimeError as error:
                logging.error(error)
            
        # start and stop recording
        if self.register_key(glfw.KEY_O, key, "Start/stop recording."):
            if not self.is_recording_frame:
                logging.info("Starting recording.")
            else:
                logging.info("Stopped recording.")
                self.save_frames()
            self.is_recording_frame = not self.is_recording_frame
            self.is_recording_display = not self.is_recording_display

        # select texture for display
        def set_display(k):
            if self.display_count >= k:
                self.display_idx = k
        
        if key is not None and key >= glfw.KEY_0 and key <= glfw.KEY_9:
            num = int(int(glfw_key(key)) + 10 * np.floor(int(glfw_key(key)) == 0)) # adds 10 only when key is 0
            set_display(num)
        
        # control modifier flags. We dont register these but these are used for activating mods
        if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            self.mods = glfw.MOD_SHIFT
        if key == glfw.KEY_LEFT_CONTROL or key == glfw.KEY_RIGHT_CONTROL:
            self.mods = glfw.MOD_CONTROL
        if key == glfw.KEY_LEFT_ALT or key == glfw.KEY_RIGHT_ALT:
            self.mods = glfw.MOD_ALT
        if key == glfw.KEY_CAPS_LOCK:
            self.mods = glfw.MOD_CAPS_LOCK
        if key == glfw.KEY_NUM_LOCK:
            self.mods = glfw.MOD_NUM_LOCK


    def key_release(self, key):
        pass

    #----------------------------------

    # handle mouse click inputs
    def mouse_click_callback(self, window, button, action, mods):
        
        if action == glfw.PRESS:
            self.mouse_is_clicked = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_down = True
                self.left_mouse_click()
            if button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_down = True
                self.right_mouse_click()    
            if button == glfw.MOUSE_BUTTON_MIDDLE:
                self.middle_mouse_down = True
                self.middle_mouse_click()    
        elif action == glfw.RELEASE:
            self.mouse_is_clicked = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.left_mouse_down = False
                self.left_mouse_release()
            if button == glfw.MOUSE_BUTTON_RIGHT:
                self.right_mouse_down = False
                self.right_mouse_release()    
            if button == glfw.MOUSE_BUTTON_MIDDLE:
                self.middle_mouse_down = False
                self.middle_mouse_release()    
            
        self.click_position = np.array(glfw.get_cursor_pos(window))

    #----------------------------------

    # overwrite these functions with custom mouse events
    def left_mouse_click(self):
        pass

    def right_mouse_click(self):
        pass

    def middle_mouse_click(self):
        pass

    def left_mouse_release(self):
        pass

    def right_mouse_release(self):
        pass

    def middle_mouse_release(self):
        pass

    #----------------------------------

    # handle mouse move inputs
    def mouse_move_callback(self, window, xpos, ypos):
        self.mouse_is_moved = True
        move_position = np.array(glfw.get_cursor_pos(window))
        self.mouse_move(move_position)
        self.need_update = True
        

    #----------------------------------

    # overwrite this function with custom mouse move event
    def mouse_move(self, move_position):
        pass

    #----------------------------------

    # handle mouse scroll input
    def mouse_scroll_callback(self, window, x_offset, y_offset):
        self.mouse_scroll(y_offset)
        self.need_update = True

    #----------------------------------

    # overwrite this function with custom mouse scroll event
    def mouse_scroll(self, sign):
        pass




        
#============================================================================


# An interactive OpenGL window with simple 2D transforms from mouse input.
# Drag left mouse to move, use mouse wheel to zoom.
class Transform2DWindow(OpenGLWindow):

    def __init__(self, display_res=(1,1), title="Transform2D Window", drag_speed=1., zoom_speed=0.05, logging_level=logging.INFO, max_fps=None):
        super().__init__(display_res, title, logging_level, max_fps)
        self.drag_speed = drag_speed
        self.zoom_speed = zoom_speed
        self.reset_transform()

        self.is_recording_transformation = False
        self.is_matrixfile_loaded = False
        self.transformations = None
        self.transformations_buffer = []



    #----------------------------------

    def reset_transform(self):
        self.scale = 1.
        self.base_position = np.zeros(2, dtype=np.float32)
        self.current_position = self.base_position

    #----------------------------------

    def post_render(self):
        if self.is_recording_transformation:
            self.transformations_buffer.append(self.transform.flatten())
            self.frame_count += 1
            self.need_update = True

        if self.is_matrixfile_loaded:
            self.need_update = True
            if self.frame_count >= self.transformations.shape[0]:
                self.save_recording()
    
    #----------------------------------

    @property
    def transform(self):
        if not self.is_matrixfile_loaded:
            translation = translation_matrix_2D(self.current_position)
            scaling = isotropic_scaling_matrix_2D(self.scale)
            return translation @ scaling
        else:
            flat_transform = np.float32(self.transformations[self.frame_count])
            tmatrix = flat_transform.reshape(3,3)
            self.scale = tmatrix[0, 0]
            self.current_position = tmatrix[0:2, 2]
            return tmatrix


    #----------------------------------

    def mouse_move(self, move_position):
        if self.left_mouse_down:
            offset = (self.click_position - move_position) * self.drag_speed
            self.current_position = self.base_position + offset / self.display_res * self.scale

    #----------------------------------

    def left_mouse_release(self):
        self.base_position = self.current_position

    #----------------------------------

    def mouse_scroll(self, sign):
        self.scale *= 1 - sign * self.zoom_speed

    #----------------------------------

    def key_press(self, key):
        if self.register_key(glfw.KEY_X, key, "Reset transform."):
            self.reset_transform()            

        # loading matrices
        if self.register_key(glfw.KEY_L, key, "Load & Replay transformations."):
            root = tk.Tk()
            root.withdraw()
            file_name = filedialog.askopenfilename() 
            if file_name:
                self.transformations = np.loadtxt(file_name, delimiter=' ')
                if not self.is_matrixfile_loaded and not self.is_recording_frame:
                    logging.info("Replaying transformations.")
                self.is_recording_frame = not self.is_recording_frame
                self.is_matrixfile_loaded = not self.is_matrixfile_loaded
            else:
                logging.info("No file Loaded.")

        # recording/saving matrices
        if self.register_key(glfw.KEY_M, key, "Record & Save transformations."):
            if not self.is_recording_transformation:
                logging.info("Recording transformation matrices.")
            else:
                logging.info("Stopped recording matrices.")
                root = tk.Tk()
                root.withdraw()
                file_name = filedialog.asksaveasfilename(
                    defaultextension = '.txt',
                    filetypes = (('Text files', '*.txt'), ('All files', '*.*'))
                    ) 
                if file_name:
                    for idx, matrix in enumerate(self.transformations_buffer):
                        print_same_line(f"Saving matrix {idx+1}/{len(self.transformations_buffer)} to disk.")
                        with open(file_name, "a") as f:
                            np.savetxt(f, [matrix], delimiter=" ", fmt='%.18e')
                    print("")
                self.transformations_buffer = []
            self.is_recording_display = not self.is_recording_display
            self.is_recording_transformation = not self.is_recording_transformation
            self.frame_count = 0

    #----------------------------------

    def save_recording(self):
        self.save_frames()
        self.is_recording_frame = False 
        self.is_matrixfile_loaded = False
        self.transformations = None

    #----------------------------------

