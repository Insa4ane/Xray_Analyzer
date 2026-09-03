import os
class Tester:
    def __init__(self, loader):
        self.path, self.is_ready=self.check_dir(loader)

    def check_dir(self, loader):
        try:
            return loader.path, True
        except Exception as e:
            return f"Error! We cannot return a path: {e}", False

    def open_file(self, name: str, category: str) -> list | tuple:
        if name and category:
            target_dir = os.path.join(self.path, "chest_xray", name, category)
            if not os.path.exists(target_dir):
                return "We cannot find the file", False

            all_images = os.listdir(target_dir)
            return all_images

        return "Invalid arguments", False





