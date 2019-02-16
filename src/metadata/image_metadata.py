from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


class ImageMetadata():
  '''
  Extract the exif data from any image. Data includes GPS coordinates,
  Focal Length, Manufacturer, and more.
  '''
  exif_data = None
  image = None

  def __init__(self, img_path):
    try:
      self.image = Image.open(img_path)
      # print(self.image._getexif())
      self.get_exif_data()

      super(ImageMetadata, self).__init__()
    except Exception as ex:
      print(ex)
      pass

  def get_exif_data(self):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = self.image._getexif()
    if info:
      for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
          gps_data = {}
          for t in value:
            sub_decoded = GPSTAGS.get(t, t)
            gps_data[sub_decoded] = value[t]

          exif_data[decoded] = gps_data
        else:
          exif_data[decoded] = value
    self.exif_data = exif_data
    return exif_data
