"""
Example:
ld = LomotifDownloader('./downloaded_lomotifs')
video_url = 'your-url-here'
result, save_file_name = ld.download(video_url=video_url)
ld.remove_downloaded_file(save_file_name)
ld.remove_downloads_folder()
"""
import os
import ray
import sys
import logging
import traceback
import requests
import shutil

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))


class LomotifDownloader:
    def __init__(self, save_folder_directory):
        """Downloads a lomotif to a folder.

        Args:
            save_folder_directory (string): path to folder where \
            lomotifs will be stored
        """
        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")
            self.logger.info("Set up lomotif downloads directory.")
            if not os.path.exists(save_folder_directory):
                os.makedirs(save_folder_directory)
            self.logger.info("Set up lomotif downloads directory completed.")
            self.save_folder_directory = save_folder_directory
        except Exception as e:
            self.logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def download(self, video_url, lomotif_id):
        """Download from a video url.

        Args:
            video_url (string): url to lomotif
            lomotif_id (string): lomotif unique ID

        Returns:
            tuple: (True if the lomotif has been downloaded successfully else False,
            file path to the downloaded lomotif, ObjectRef of the video bytes)
        """
        self.logger.info("[{}] Attempt to download lomotif.".format(lomotif_id))
        resp = requests.get(video_url)
        self.logger.info(
            "[{}] Request status code: {}".format(lomotif_id, resp.status_code)
        )
        if resp.status_code != 403:
            video_file_name = os.path.basename(video_url)
            save_file_name = os.path.join(self.save_folder_directory, video_file_name)
            # with open(save_file_name, "wb") as f:
            #     f.write(resp.content)
            save_file_name_ref = ray.put(resp.content)
            self.logger.info("[{}] Lomotif successfully downloaded.".format(lomotif_id))
            return True, save_file_name, save_file_name_ref
        else:
            self.logger.error("[{}] Lomotif download failed.".format(lomotif_id))
            return False, None, None

    # def remove_downloaded_file(self, save_file_name, lomotif_id):
    #     if os.path.exists(save_file_name):
    #         os.remove(save_file_name)
    #         self.logger.info("[{}] Lomotif deleted from local.".format(lomotif_id))

    # def remove_downloads_folder(self):
    #     if os.path.exists(self.save_folder_directory):
    #         shutil.rmtree(self.save_folder_directory)
    #         self.logger.info("Deleted folder {}".format(self.save_folder_directory))
