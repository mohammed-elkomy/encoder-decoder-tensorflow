######################################## my drive libaray ########################################
# ----------------------------------------------------------------------------------------
# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
import os
import time
import zipfile

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class DriveRemoteSnapshots:
    # check projects directories
    def __init__(self, project_name, logger):
        # ----------------------------------------------------------------------------------------
        # Authenticate and create the PyDrive client.
        # This only needs to be done once per notebook.
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile('credentials.txt')  # we need this
        self.drive = GoogleDrive(gauth)

        self.my_projects_folder_id = "1B82anWV8Mb4iHYmOp9tIR9aOTlfllwsD"
        self.project_name = project_name
        self.project_id = self.make_sure_project()

        self.logger = logger

    def is_project_exsists(self, project_name):
        file_list = self.drive.ListFile({'q': '"{}" in parents and mimeType = "application/vnd.google-apps.folder" and trashed=false'.format(self.my_projects_folder_id)}).GetList()
        return len(list(file for file in file_list if file["title"] == project_name)) > 0

    def make_sure_project(self):
        if not self.is_project_exsists(self.project_name):
            folder_metadata = {'title': self.project_name, 'mimeType': 'application/vnd.google-apps.folder', "parents": [{"kind": "self.drive#fileLink", "id": self.my_projects_folder_id}]}
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()

        file_list = self.drive.ListFile({'q': '"{}" in parents and mimeType = "application/vnd.google-apps.folder" and trashed=false'.format(self.my_projects_folder_id)}).GetList()
        return list(file for file in file_list if file["title"] == self.project_name)[0]['id']

    def search_file(self, file_name):
        return self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType != 'application/vnd.google-apps.folder'".format(file_name)}).GetList()

    def search_folder(self, folder_name):
        return self.drive.ListFile({'q': "title='{}' and trashed=false and mimeType = 'application/vnd.google-apps.folder'".format(folder_name)}).GetList()

    def upload_file(self, file_path):
        upload_started = time.time()
        title = os.path.split(file_path)[-1]
        uploaded = self.drive.CreateFile({'title': title, "parents": [{"kind": "drive#fileLink", "id": self.project_id}]})
        uploaded.SetContentFile(file_path)  # file on disk
        uploaded.Upload()
        self.logger.log_upload_drive(uploaded.get('id'), title, upload_started)

    def list_project_files(self):
        return sorted(self.drive.ListFile({'q': "trashed=false and mimeType != 'application/vnd.google-apps.folder' and '{}' in parents".format(self.project_id)}).GetList(), key=lambda k: k['title'])

    def get_latest_snapshot_meta(self):
        if len(self.list_project_files()) > 0:
            return True, self.list_project_files()[-1]['title'], self.list_project_files()[-1]['id']
        else:
            return False, None, None

    def get_latest_snapshot(self):
        download_started = time.time()
        if_possible, save_path, file_id = self.get_latest_snapshot_meta()
        if if_possible:
            downloaded = self.drive.CreateFile({'id': file_id})
            downloaded.GetContentFile(save_path)  # Download file and save locally
            zip_ref = zipfile.ZipFile(save_path, 'r')
            zip_ref.extractall('.')
            zip_ref.close()
            self.logger.log_download_drive(downloaded['id'], downloaded['title'], download_started)

        return if_possible
