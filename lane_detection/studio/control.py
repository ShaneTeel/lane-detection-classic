import cv2
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class Controller:

    def __init__(self, source):
        self.paused = False
        self.exit = False
        self.source = source
        self.current_frame = 0
        self.last_frame = self.source.frame_count - 1 if self.source.cap is not None else None

    def playback_controls(self):
        if self.source.source_type == "image":
            cv2.waitKey(0)
            self.exit = True
            return self.exit
        else: 
            if not self.exit:
                while self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('p'), ord('P'), ord(' ')]:
                        logger.info(f"Resuming video at {self.current_frame}")
                        self.paused = False
                    elif key in [ord('q'), ord('Q'), 27]:
                        logger.info(f"Exiting video player at frame {self.current_frame}.")
                        self.exit = True
                        break
                    elif key == ord('-'):
                        self.current_frame = (self.current_frame - 50) + self.last_frame if self.current_frame - 50 <= 0 else self.current_frame - 50
                        logger.info(f"Skipping to frame {self.current_frame}")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    elif key == ord('+'):
                        self.current_frame = (self.current_frame + 50) - self.last_frame if self.current_frame + 50 > self.last_frame else self.current_frame + 50
                        logger.info(f"Skipping to frame {self.current_frame}")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    elif key in [ord('r'), ord('R')]:
                        self.current_frame = 0
                        logger.info("Restarting stream.")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

                if not self.paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('p'), ord('P'), ord(' ')]:
                        self.paused = True
                        logger.info(f"Pausing at frame {self.current_frame}.")
                    elif key in [ord('q'), ord('Q'), 27]:
                        logger.info(f"Exiting video player at frame {self.current_frame}.")
                        self.exit = True
                    elif key == ord('-'):
                        self.current_frame = (self.current_frame - 50) + self.last_frame if self.current_frame - 50 <= 0 else self.current_frame - 50
                        logger.info(f"Skipping to frame {self.current_frame}")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    elif key == ord('+'):
                        self.current_frame = (self.current_frame + 50) - self.last_frame if self.current_frame + 50 > self.last_frame else self.current_frame + 50
                        logger.info(f"Skipping to frame {self.current_frame}")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    elif key in [ord('r'), ord('R')]:
                        self.current_frame = 0
                        logger.info(f"Restarting stream.")
                        self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    
                    self.current_frame += 1
            else:
                return self.exit
        
    def print_playback_menu(self):

        print("-------------------------------------------")
        print("\t\033[1;4mPlayback Controls\033[0m\n")
        print("     \033[1mCommand      | Wait Key\033[0m")
        print("     --------------------------")
        print("     \033[3mQuit\033[0m         : 'q', 'Q' or ESC")
        print("     \033[3mPause/Resume\033[0m : 'p', 'P', or SPACE")
        print("     \033[3mFast-Forward\033[0m : '+'")
        print("     \033[3mRewind\033[0m       : '-'")
        print("     \033[3mRestart\033[0m      : 'r', 'R'")
        print("------------------------------------------")