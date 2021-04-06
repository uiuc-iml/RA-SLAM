import os
import csv
import urllib
import urllib.request

scannetv2_label_name = "scannetv2-labels.combined.tsv"
scannetv2_label_path = "http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannetv2-labels.combined.tsv"

def download_if_not_exists(file_path, url = scannetv2_label_path):
    """Given a file path and an url, download if the file does not exist"""
    if not os.path.isfile(file_path):
        print("Label file not found. Downloading to {}".format(file_path))
        urllib.request.urlretrieve(url, file_path)

NYU40_HT_DICT = {
    'wall': 0,
    'bookshelf': 1,
    'picture': 0,
    'counter': 1,
    'blinds': 0,
    'desk': 1,
    'shelves': 1,
    'curtain': 1,
    'dresser': 1,
    'pillow': 1,
    'mirror': 0,
    'floor': 0,
    'floor mat': 1,
    'clothes': 0,
    'ceiling': 0,
    'books': 1,
    'refridgerator': 1,
    'television': 0,
    'paper': 0,
    'towel': 1,
    'shower curtain': 1,
    'box': 1,
    'cabinet': 1,
    'whiteboard': 0,
    'person': 0,
    'night stand': 1,
    'toilet': 1,
    'sink': 1,
    'lamp': 1,
    'bathtub': 1,
    'bag': 0,
    'otherstructure': 1,
    'otherfurniture': 1,
    'bed': 1,
    'otherprop': 1,
    'chair': 1,
    'sofa': 1,
    'table': 1,
    'door': 1,
    'window': 0
}

class LabelParser:
    """
    Generate label map for scannet v2 labels
    """
    def __init__(self, root_dir = "/data/"):
        """
        Parse scannet provided label file using CSV lib
        """
        self.root_dir = root_dir
        self.scannet_label_path = os.path.join(self.root_dir, scannetv2_label_name)
        download_if_not_exists(self.scannet_label_path)

        # self.nyu40_dict: nyu40 id -> nyu40class name
        # note that nyu40 id is the label property in ScanNetv2
        # annotated triangle mesh file
        self.nyu40_dict = {}

        with open(self.scannet_label_path, newline = '') as csvfile:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                if int(row["nyu40id"]) in self.nyu40_dict:
                    assert self.nyu40_dict[int(row["nyu40id"])] == row["nyu40class"]
                else:
                    self.nyu40_dict[int(row["nyu40id"])] = row["nyu40class"]

    def get_nyuid_to_nyuclass_map(self):
        """
        return: dict whose key is nyu40id and value is name oo nyu40class
        """
        return self.nyu40_dict
    
    def get_nyuid_to_ht_map(self):
        """
        return: dict whose key is nyu40id and value is 1/0 (high-touch/low-touch)
        """
        ret = {}
        for k in self.nyu40_dict:
            ret[int(k)] = NYU40_HT_DICT[self.nyu40_dict[k]]
        return ret

if __name__ == '__main__':
    my_parser = LabelParser()
    print(my_parser.nyu40_dict)
    print(repr(my_parser.get_nyuid_to_ht_map()))
